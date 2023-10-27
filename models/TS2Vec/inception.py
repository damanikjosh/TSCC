import torch
import torch.nn as nn


def correct_sizes(sizes):
    corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
    return corrected_sizes


def pass_through(X):
    return X


class Inception(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(),
                 return_indices=False):
        """
        : param in_channels				Number of input channels (input features)
        : param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
        : param kernel_sizes			List of kernel sizes for each convolution.
                                        Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                        This is nessesery because of padding size.
                                        For correction of kernel_sizes use function "correct_sizes".
        : param bottleneck_channels		Number of output channels in bottleneck.
                                        Bottleneck wont be used if nuber of in_channels is equal to 1.
        : param activation				Activation function for output tensor (nn.ReLU()).
        : param return_indices			Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d.
        """
        super(Inception, self).__init__()
        self.return_indices = return_indices
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

        self.conv_from_bottleneck_1 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False
        )
        self.conv_from_bottleneck_2 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False
        )
        self.conv_from_bottleneck_3 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False
        )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
        self.conv_from_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(num_features=4 * n_filters)
        self.activation = activation

    def forward(self, X):
        # X: (batch_size, in_channels, seq_len)
        # step 1
        Z_bottleneck = self.bottleneck(X)
        if self.return_indices:
            Z_maxpool, indices = self.max_pool(X)
        else:
            Z_maxpool = self.max_pool(X)
        # step 2
        Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
        Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
        Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
        Z4 = self.conv_from_maxpool(Z_maxpool)
        # step 3
        Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
        Z = self.activation(self.batch_norm(Z))
        if self.return_indices:
            return Z, indices
        else:
            return Z


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=None, bottleneck_channels=32, use_residual=True,
                 activation=nn.ReLU(), return_indices=False):
        super(InceptionBlock, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [9, 19, 39]
        self.use_residual = use_residual
        self.return_indices = return_indices
        self.activation = activation
        self.inception_1 = Inception(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_2 = Inception(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_3 = Inception(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=4 * n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(
                    num_features=4 * n_filters
                )
            )

    def forward(self, X):
        if self.return_indices:
            Z, i1 = self.inception_1(X)
            Z, i2 = self.inception_2(Z)
            Z, i3 = self.inception_3(Z)
        else:
            Z = self.inception_1(X)
            Z = self.inception_2(Z)
            Z = self.inception_3(Z)
        if self.use_residual:
            Z = Z + self.residual(X)
            Z = self.activation(Z)
        if self.return_indices:
            return Z, [i1, i2, i3]
        else:
            return Z


class InceptionTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32,
                 activation=nn.ReLU()):
        """
        : param in_channels				Number of input channels (input features)
        : param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
        : param kernel_sizes			List of kernel sizes for each convolution.
                                        Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                        This is nessesery because of padding size.
                                        For correction of kernel_sizes use function "correct_sizes".
        : param bottleneck_channels		Number of output channels in bottleneck.
                                        Bottleneck wont be used if nuber of in_channels is equal to 1.
        : param activation				Activation function for output tensor (nn.ReLU()).
        """
        super(InceptionTranspose, self).__init__()
        self.activation = activation
        self.conv_to_bottleneck_1 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False
        )
        self.conv_to_bottleneck_2 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False
        )
        self.conv_to_bottleneck_3 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False
        )
        self.conv_to_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.max_unpool = nn.MaxUnpool1d(kernel_size=3, stride=1, padding=1)
        self.bottleneck = nn.Conv1d(
            in_channels=3 * bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X, indices):
        Z1 = self.conv_to_bottleneck_1(X)
        Z2 = self.conv_to_bottleneck_2(X)
        Z3 = self.conv_to_bottleneck_3(X)
        Z4 = self.conv_to_maxpool(X)

        Z = torch.cat([Z1, Z2, Z3], axis=1)
        MUP = self.max_unpool(Z4, indices)
        BN = self.bottleneck(Z)
        # another possibility insted of sum BN and MUP is adding 2nd bottleneck transposed convolution

        return self.activation(self.batch_norm(BN + MUP))


class InceptionTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32,
                 use_residual=True, activation=nn.ReLU()):
        super(InceptionTransposeBlock, self).__init__()
        self.use_residual = use_residual
        self.activation = activation
        self.inception_1 = InceptionTranspose(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        self.inception_2 = InceptionTranspose(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        self.inception_3 = InceptionTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(
                    num_features=out_channels
                )
            )

    def forward(self, X, indices):
        assert len(indices) == 3
        Z = self.inception_1(X, indices[2])
        Z = self.inception_2(Z, indices[1])
        Z = self.inception_3(Z, indices[0])
        if self.use_residual:
            Z = Z + self.residual(X)
            Z = self.activation(Z)
        return Z


class InceptionTime(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(),
                 use_residual=True, return_indices=False):
        super(InceptionTime, self).__init__()
        self.return_indices = return_indices
        self.use_residual = use_residual
        self.activation = activation
        self.inception_block_1 = InceptionBlock(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_block_2 = InceptionBlock(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_block_3 = InceptionBlock(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_block_4 = InceptionBlock(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_block_5 = InceptionBlock(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_block_6 = InceptionBlock(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
            activation=activation,
            return_indices=return_indices
        )

    def forward(self, X):
        if self.return_indices:
            Z, i1 = self.inception_block_1(X)
            Z, i2 = self.inception_block_2(Z)
            Z, i3 = self.inception_block_3(Z)
            Z, i4 = self.inception_block_4(Z)
            Z, i5 = self.inception_block_5(Z)
            Z, i6 = self.inception_block_6(Z)
        else:
            Z = self.inception_block_1(X)
            Z = self.inception_block_2(Z)
            Z = self.inception_block_3(Z)
            Z = self.inception_block_4(Z)
            Z = self.inception_block_5(Z)
            Z = self.inception_block_6(Z)
        if self.return_indices:
            return Z, [i1, i2, i3, i4, i5, i6]
        else:
            return Z

# class Flatten(nn.Module):
#     def __init__(self, out_features):
#         super(Flatten, self).__init__()
#         self.output_dim = out_features
#
#     def forward(self, x):
#         return x.view(-1, self.output_dim)
#
#
# class Reshape(nn.Module):
#     def __init__(self, out_shape):
#         super(Reshape, self).__init__()
#         self.out_shape = out_shape
#
#     def forward(self, x):
#         return x.view(-1, *self.out_shape)
#
#
# class Transpose(nn.Module):
#     def __init__(self, dim1, dim2):
#         super(Transpose, self).__init__()
#         self.dim1 = dim1
#         self.dim2 = dim2
#
#     def forward(self, x):
#         return x.transpose(self.dim1, self.dim2)
#
#
# # Given a sequence of length 2500 with input dimension 3
#
# class InceptionTime(nn.Module):
#     def __init__(self, in_channels=1, n_filters=32, num_layers=2, kernel_sizes=[9, 19, 39], latent_size=10, bottleneck_channels=32, activation=nn.ReLU(), use_residual=True):
#         super(InceptionTime, self).__init__()
#         self.transpose = Transpose(1, 2)
#         self.inception_blocks = nn.ModuleList()
#         for layer in range(num_layers):
#             self.inception_blocks.append(
#                 InceptionBlock(
#                     in_channels=in_channels,
#                     n_filters=n_filters,
#                     kernel_sizes=kernel_sizes,
#                     bottleneck_channels=bottleneck_channels,
#                     activation=activation,
#                     use_residual=use_residual
#                 )
#             )
#             in_channels = n_filters * 4
#         self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
#         self.flatten = Flatten(out_features=n_filters * 4)
#         self.fc = nn.Linear(n_filters * 4, latent_size)
#         self.activation = activation
#
#     def forward(self, x):
#         x = self.transpose(x)
#         for inception_block in self.inception_blocks:
#             x = inception_block(x)
#         x = self.avg_pool(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x