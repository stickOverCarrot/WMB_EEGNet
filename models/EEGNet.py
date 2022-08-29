from .networks import *


class EEGNet(BaseModel):
    def __init__(self,
                 in_chans,
                 input_time_length,
                 n_classes,
                 pool_mode='mean',
                 f1=8,
                 d=2,
                 f2=16,
                 kernel_length=64,
                 drop_prob=0.25,
                 ):
        super(EEGNet, self).__init__()

        # Assigns all parameters in init to self.param_name
        self.__dict__.update(locals())
        del self.self

        # Define kind of pooling used:
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.temporal_conv = nn.Sequential(
            Expression(_transpose_to_b_1_c_0),
            nn.Conv2d(in_channels=1, out_channels=self.f1,
                      kernel_size=(1, self.kernel_length),
                      stride=1,
                      bias=False,
                      padding=(0, self.kernel_length // 2)),
            nn.BatchNorm2d(self.f1, momentum=0.01, affine=True, eps=1e-3)
        )

        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(self.f1, self.f1 * self.d, (self.in_chans, 1),
                                 max_norm=1, stride=1, bias=False,
                                 groups=self.f1, padding=(0, 0)),
            # nn.Conv2d(self.f1, self.f1 * self.d, (self.in_chans, 1), stride=1, bias=False,
            #           groups=self.f1, padding=(0, 0)),
            nn.BatchNorm2d(self.f1 * self.d, momentum=0.01, affine=True,
                           eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        )

        self.separable_conv = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(self.f1 * self.d, self.f1 * self.d, (1, 16), stride=1,
                      bias=False, groups=self.f1 * self.d,
                      padding=(0, 16 // 2)),
            nn.Conv2d(self.f1 * self.d, self.f2, (1, 1), stride=1, bias=False,
                      padding=(0, 0)),
            nn.BatchNorm2d(self.f2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        )

        out = np_to_var(
            np.ones((1, self.in_chans, self.input_time_length, 1),
                    dtype=np.float32))
        out = self.forward_init(out)
        # out = self.separable_conv(self.spatial_conv(self.temporal_conv(out)))
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.cls = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            Conv2dWithConstraint(self.f2, self.n_classes,
                                 (n_out_virtual_chans, self.final_conv_length), max_norm=0.5,
                                 bias=True),
            # nn.Conv2d(self.f2, self.n_classes,
            #          (n_out_virtual_chans, self.final_conv_length), bias=True),
            Expression(_transpose_1_0),
            Expression(_squeeze_final_output),
        )

        self.apply(glorot_weight_zero_bias)

    def forward_init(self, x):
        with th.no_grad():
            for module in self._modules:
                if isinstance(self._modules[module], th.nn.ModuleList):
                    x = self._modules[module][0](x)
                else:
                    x = self._modules[module](x)
        return x

    def forward(self, x):
        x = x[:, :, :, None]
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        feats = self.separable_conv(x)
        x = self.cls(feats)
        return x, feats


class WMB_EEGNet(BaseModel):
    def __init__(self,
                 in_chans,
                 input_time_length,
                 n_classes,
                 pool_mode='mean',
                 f1=8,
                 d=2,
                 f2=16,
                 kernel_length=64,
                 drop_prob=0.25,
                 source_num=8,
                 ):
        super(WMB_EEGNet, self).__init__()

        # Assigns all parameters in init to self.param_name
        self.__dict__.update(locals())
        del self.self

        # Define kind of pooling used:
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.temporal_conv = nn.Sequential(
            Expression(_transpose_to_b_1_c_0),
            nn.Conv2d(in_channels=1, out_channels=self.f1,
                      kernel_size=(1, self.kernel_length),
                      stride=1,
                      bias=False,
                      padding=(0, self.kernel_length // 2)),
            nn.BatchNorm2d(self.f1, momentum=0.01, affine=True, eps=1e-3)
        )

        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(self.f1, self.f1 * self.d, (self.in_chans, 1),
                                 max_norm=1, stride=1, bias=False,
                                 groups=self.f1, padding=(0, 0)),
            nn.BatchNorm2d(self.f1 * self.d, momentum=0.01, affine=True,
                           eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        )

        self.separable_conv = nn.ModuleList(nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(self.f1 * self.d, self.f1 * self.d, (1, 16), stride=1,
                      bias=False, groups=self.f1 * self.d,
                      padding=(0, 16 // 2)),
            nn.Conv2d(self.f1 * self.d, self.f2, (1, 1), stride=1, bias=False,
                      padding=(0, 0)),
            nn.BatchNorm2d(self.f2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        ) for i in range(self.source_num))

        out = np_to_var(
            np.ones((1, self.in_chans, self.input_time_length, 1),
                    dtype=np.float32))
        out = self.forward_init(out)
        # out = self.separable_conv(self.spatial_conv(self.temporal_conv(out)))
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.cls = nn.ModuleList(nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            Conv2dWithConstraint(self.f2, self.n_classes,
                                 (n_out_virtual_chans, self.final_conv_length), max_norm=0.5,
                                 bias=True),
            Expression(_transpose_1_0),
            Expression(_squeeze_final_output),
        ) for i in range(self.source_num))

        self.apply(glorot_weight_zero_bias)
        self.weight = nn.Parameter(th.FloatTensor([1. / self.source_num] * self.source_num), requires_grad=True)

    def forward_init(self, x):
        with th.no_grad():
            for module in self._modules:
                if isinstance(self._modules[module], th.nn.ModuleList):
                    x = self._modules[module][0](x)
                else:
                    x = self._modules[module](x)
        return x

    def feature_extractor(self, x):
        x = x[:, :, :, None]
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        return x

    def forward_target_source(self, x):
        batch_size = x.shape[0]
        assert 0 == batch_size % (self.source_num + 1)
        batch_size_s = batch_size // (self.source_num + 1)
        source_cls = []
        target_cls = []
        x = self.feature_extractor(x)
        for i in range(self.source_num):
            feed_together = th.cat([x[i * batch_size_s:(i + 1) * batch_size_s], x[self.source_num * batch_size_s:]])
            output = self.separable_conv[i](feed_together)
            source_cls.append(self.cls[i](output[:batch_size_s]))
            target_cls.append(self.cls[i](output[batch_size_s:]))
            # source_feats.append(output[:batch_size_s])
            # target_feats.append(output[batch_size_s:])
            # source_feats.append(self.separable_conv[i](x[i * batch_size_s:(i + 1) * batch_size_s]))
            # target_feats.append(self.separable_conv[i](x[self.source_num * batch_size_s:]))
        # for i in range(self.source_num):
        #     source_cls.append(self.cls[i](source_feats[i]))
        #     target_cls.append(self.cls[i](target_feats[i]))
        all_cls = th.stack(target_cls, dim=1)
        cls = F.softmax(all_cls, dim=-1) * F.softmax(self.weight, dim=-1).view(1, -1, 1)
        cls = th.sum(cls, dim=1)
        return source_cls, target_cls, cls

    def forward_target_only(self, x, c=None):
        x = self.feature_extractor(x)
        target_feats = []
        target_cls = []
        for i in range(self.source_num):
            target_feats.append(self.separable_conv[i](x))
        for i in range(self.source_num):
            target_cls.append(self.cls[i](target_feats[i]))
        if c is not None:
            if isinstance(c, list):
                target_cls = [target_cls[i] for i in c]
                weight = self.weight[c]
            else:
                target_cls = target_cls[c]
                cls = F.softmax(target_cls, dim=-1)
                return target_feats, target_cls, cls
        else:
            weight = self.weight
        all_cls = th.stack(target_cls, dim=1)
        cls = F.softmax(all_cls, dim=-1) * F.softmax(weight, dim=-1).view(1, -1, 1)
        cls = th.sum(cls, dim=1)
        return target_feats, target_cls, cls

    def forward(self, x, is_target_only=True, c=None):
        if not is_target_only:
            return self.forward_target_source(x)
        return self.forward_target_only(x, c)


def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)


def _transpose_1_0(x):
    return x.permute(0, 1, 3, 2)


def _review(x):
    return x.contiguous().view(-1, x.size(2), x.size(3))


def _squeeze_final_output(x):
    """
    Remove empty dim at end and potentially remove empty time dim
    Do not just use squeeze as we never want to remove first dim
    :param x:
    :return:
    """
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x
