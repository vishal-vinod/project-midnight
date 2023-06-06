import torch
import torch.nn as nn
import torch.nn.functional as F


class lReLU(nn.Module):
    def __init__(self):
        super(lReLU, self).__init__()

    def forward(self, x):
        return torch.max(x * 0.2, x)


class Double_Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Double_Conv2d, self).__init__()
        self.double_conv2d = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
            ),
            lReLU(),
            # nn.InstanceNorm2d(out_channel, affine=True)
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
            ),
            lReLU()
            # nn.InstanceNorm2d(out_channel, affine=True)
        )

    def forward(self, x):
        return self.double_conv2d(x)

class JPEGCompress(nn.Module):
    def __init__(self):
        super(JPEGCompress, self).__init__()
        self.conv0 = nn.Conv2d(12, 4, 3, padding=1)
        self.conv1 = Double_Conv2d(4, 64)
        self.conv2 = Double_Conv2d(64, 128)
        self.conv3 = Double_Conv2d(128, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = Double_Conv2d(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = Double_Conv2d(128, 12)

    def forward(self, x):
        dc = self.conv0(x)
#         x = F.pixel_shuffle(x, 2)
        conv1 = self.conv1(dc)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)

        up8 = self.up8(conv3)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)

        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)

        out = F.pixel_shuffle(conv9, 2)

        return dc, out


class UNetSony(nn.Module):
    def __init__(self):
        super(UNetSony, self).__init__()
        self.conv1 = Double_Conv2d(64, 64)
        self.conv2 = Double_Conv2d(64, 128)
        self.conv3 = Double_Conv2d(128, 256)
        self.conv4 = Double_Conv2d(256, 512)
        self.conv5 = Double_Conv2d(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = Double_Conv2d(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = Double_Conv2d(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = Double_Conv2d(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = Double_Conv2d(128, 64)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3, kernel_size=2)

        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4, kernel_size=2)

        conv5 = self.conv5(pool4)

        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)

        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)

        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)

        return conv9


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Encoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel, out_channels=16, kernel_size=3, padding=1
            ),
            lReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            lReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=out_channel, kernel_size=3, padding=1
            ),
            lReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decoder_1, self).__init__()
        self.decoder = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel, out_channels=32, kernel_size=3, padding=1
            ),
            lReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            lReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_channel, kernel_size=1),
            lReLU(),
        )

    def forward(self, x):

        dc = self.decoder(x)

        return dc

    
class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel, out_channels=32, kernel_size=3, padding=1
            ),
            lReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            lReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_channel, kernel_size=1),
            lReLU(),
        )

    def forward(self, x):

        dc = self.decoder(x)
        out = F.pixel_shuffle(dc, 2)

        return out


class Decoder_RAW(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decoder_RAW, self).__init__()
        self.decoder = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel, out_channels=32, kernel_size=3, padding=1
            ),
            lReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            lReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_channel, kernel_size=1),
            lReLU(),
        )

    def forward(self, x):

        dc = self.decoder(x)

        return dc


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


class Task_filter(nn.Module):
    def __init__(self):
        super(Task_filter, self).__init__()
        self.en_s = Encoder(4, 64)
        self.en_t = Encoder(4, 64)
        self.unet = UNetSony()
        self.dc = Decoder(64, 12)

    def forward(self, x, source):

        if source:

            for param in self.en_s.parameters():
                param.requires_grad = True
            for param in self.en_t.parameters():
                param.requires_grad = False
            for param in self.unet.parameters():
                param.requires_grad = True
            for param in self.dc.parameters():
                param.requires_grad = True

            en = self.en_s(x)
            unet = self.unet(en)
            dc = self.dc(unet)

            return dc

        else:

            for param in self.en_s.parameters():
                param.requires_grad = False
            for param in self.en_t.parameters():
                param.requires_grad = True
            for param in self.unet.parameters():
                param.requires_grad = True
            for param in self.dc.parameters():
                param.requires_grad = True

            en = self.en_t(x)
            unet = self.unet(en)
            dc = self.dc(unet)

            return dc

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

                
class Task_filter_RAW(nn.Module):
    def __init__(self):
        super(Task_filter_RAW, self).__init__()
        self.en_s = Encoder(4, 64)
        self.en_t = Encoder(4, 64)
        self.unet = UNetSony()
        self.dc = Decoder_1(64, 12)
        self.compress = JPEGCompress()

    def forward(self, x, source):

        if source:
            for param in self.en_s.parameters():
                param.requires_grad = True
            for param in self.en_t.parameters():
                param.requires_grad = False
            for param in self.unet.parameters():
                param.requires_grad = True
            for param in self.dc.parameters():
                param.requires_grad = True

            en = self.en_s(x)
            unet = self.unet(en)
            dc = self.dc(unet)
            dc = F.pixel_shuffle(dc, 2)

            return dc, None

        else:

            for param in self.en_s.parameters():
                param.requires_grad = False
            for param in self.en_t.parameters():
                param.requires_grad = True
            for param in self.unet.parameters():
                param.requires_grad = True
            for param in self.dc.parameters():
                param.requires_grad = True

            en = self.en_t(x)
            unet = self.unet(en)
            dc = self.dc(unet)
            
            dc1, out_2 = self.compress(dc)

            return dc1, out_2

        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)           


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.double_conv2d = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
            ),
            lReLU(),
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
            ),
            lReLU(),
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
            ),
            lReLU(),
        )

    def forward(self, x):
        return self.double_conv2d(x)


class Unprocess(nn.Module):
    def __init__(self):
        super(Unprocess, self).__init__()
        self.conv1 = ConvBlock(4, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.conv5 = ConvBlock(256, 512)

        self.up6 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.convUp6 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv6 = ConvBlock(512, 256)
        self.up7 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.convUp7 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv7 = ConvBlock(256, 128)
        self.up8 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.convUp8 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv8 = ConvBlock(128, 64)
        self.up9 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.convUp9 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv9 = ConvBlock(64, 32)

        self.conv10 = nn.Conv2d(32, 12, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3, kernel_size=2)

        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4, kernel_size=2)

        conv5 = self.conv5(pool4)  # 512

        up6 = self.up6(conv5)  # 512
        up6 = self.convUp6(up6)  # 256
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)  # 256

        up7 = self.up7(conv6)  # 256
        up7 = self.convUp7(up7)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)

        up8 = self.up8(conv7)
        up8 = self.convUp8(up8)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)

        up9 = self.up9(conv8)
        up9 = self.convUp9(up9)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)

        conv10 = self.conv10(conv9)
        out = F.pixel_shuffle(conv10, 2)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


def activation(output_layer):
    """Domain Transformation"""
    return torch.log(torch.pow(output_layer / 255.0, 2.0) + 1.0 / 255.0)


class HDRCNN(nn.Module):
    def __init__(self):
        super(HDRCNN, self).__init__()
        self.conv1 = Double_Conv2d(4, 64)
        self.conv2 = Double_Conv2d(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.conv5 = ConvBlock(512, 512)

        # ------------------------------------------

        self.mid_conv1 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.bn = nn.BatchNorm2d(512)
        self.activ = lReLU()

        # ------------------------------------------

        self.up6 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv6 = ConvBlock(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = ConvBlock(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = ConvBlock(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = ConvBlock(128, 64)

        self.conv10 = nn.Conv2d(64, 3, kernel_size=1)
        self.bn10 = nn.BatchNorm2d(3)
        self.activ10 = lReLU()

        self.up10 = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3, kernel_size=2)

        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4, kernel_size=2)

        conv5 = self.conv5(pool4)  # 512

        # -------------------------------------

        mid_conv = self.mid_conv1(conv5)
        mid_conv = self.bn(mid_conv)
        mid_conv = self.activ(mid_conv)

        # -------------------------------------
        up6 = self.up6(conv5)  # 512
        conv4 = activation(conv4)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)  # 256

        up7 = self.up7(conv6)  # 256
        conv3 = activation(conv3)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)

        up8 = self.up8(conv7)
        conv2 = activation(conv2)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)

        up9 = self.up9(conv8)
        conv1 = activation(conv1)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)

        conv10 = self.conv10(conv9)
        conv10 = self.bn10(conv10)
        conv10 = self.activ10(conv10)
        # out = F.pixel_shuffle(conv10, 2)
        out = self.up10(conv10)

        return out


class SSLEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSLEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel, out_channels=16, kernel_size=3, padding=1
            ),
            lReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            lReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=out_channel, kernel_size=3, padding=1
            ),
            lReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class Task_SSL1(nn.Module):
    def __init__(self):
        super(Task_SSL1, self).__init__()
        self.encoder = SSLEncoder(4, 3)
        self.conv0 = nn.Conv2d(4, 32, 3, 1, padding=1)
        self.conv = nn.Conv2d(4, 64, 9, 1, padding=4)
        self.conv1 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2)
        self.conv5 = nn.Conv2d(128, 64, 3, 1, padding=1)
        self.conv7 = nn.Conv2d(96, 64, 3, 1, padding=1)
        self.conv8 = nn.Conv2d(64, 16, 3, 1, padding=1)

    def forward(self, x):
        x = self.encoder(x)
        x_enc = x
        x_max = torch.max(x, dim=1, keepdim=False)[0]
        x_max = torch.unsqueeze(x_max, 0)
        x = torch.cat([x_max, x], 1)
        z0 = self.conv0(x)
        z0 = F.relu(z0, inplace=True)
        z = self.conv(x)
        z1 = self.conv1(z)
        z1 = F.relu(z1, inplace=True)
        z2 = self.conv2(z1)
        z2 = F.relu(z2, inplace=True)
        z3 = self.conv3(z2)
        z3 = F.relu(z3, inplace=True)
        z4 = self.conv4(z3)
        z4 = F.relu(z4, inplace=True)
        con4_ba2 = torch.cat([z4, z1], 1)
        z5 = self.conv5(con4_ba2)
        z5 = F.relu(z5, inplace=True)
        conv6 = torch.cat([z5, z0], 1)
        z7 = self.conv7(conv6)
        z8 = self.conv8(z7)
        z8 = F.pixel_shuffle(z8, 2)
        R = F.sigmoid(z8[:, 0:3, :, :])
        L = F.sigmoid(z8[:, 3:4, :, :])

        # print(f'Shape of R: {R.shape}\nShape of L: {L.shape}')

        return R, L, x_enc

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


class Task_filter_RAW_2(nn.Module):
    def __init__(self):
        super(Task_filter_RAW_2, self).__init__()
        self.en_s = Encoder(4, 64)
        self.en_t = Encoder(4, 64)
        self.unet = UNetSony()
        self.dc = Decoder_RAW(64, 4)

    def forward(self, x, source):

        if source:

            for param in self.en_s.parameters():
                param.requires_grad = True
            for param in self.en_t.parameters():
                param.requires_grad = False
            for param in self.unet.parameters():
                param.requires_grad = True
            for param in self.dc.parameters():
                param.requires_grad = True

            en = self.en_s(x)
            unet = self.unet(en)
            dc = self.dc(unet)

            return dc

        else:

            for param in self.en_s.parameters():
                param.requires_grad = False
            for param in self.en_t.parameters():
                param.requires_grad = True
            for param in self.unet.parameters():
                param.requires_grad = True
            for param in self.dc.parameters():
                param.requires_grad = True

            en = self.en_t(x)
            unet = self.unet(en)
            dc = self.dc(unet)

            return dc

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


class Task_filter_PGT(nn.Module):
    def __init__(self):
        super(Task_filter_PGT, self).__init__()
        self.en_s = Encoder(4, 64)
        self.en_t = Encoder(4, 64)
        self.en_pgt = Encoder(4, 64)
        self.unet = UNetSony()
        self.dc = Decoder(64, 12)
        self.dc_raw = Decoder_RAW(64, 4)

    def forward(self, x, source):
        if source == "Source":
            for param in self.en_s.parameters():
                param.requires_grad = True
            for param in self.en_t.parameters():
                param.requires_grad = False
            for param in self.en_pgt.parameters():
                param.requires_grad = False
            for param in self.unet.parameters():
                param.requires_grad = True
            for param in self.dc.parameters():
                param.requires_grad = True

            en = self.en_s(x)
            unet = self.unet(en)
            dc = self.dc(unet)

            return dc

        elif source == "TargetGT":
            for param in self.en_s.parameters():
                param.requires_grad = False
            for param in self.en_t.parameters():
                param.requires_grad = True
            for param in self.en_pgt.parameters():
                param.requires_grad = False
            for param in self.unet.parameters():
                param.requires_grad = True
            for param in self.dc.parameters():
                param.requires_grad = True

            en = self.en_t(x)
            unet = self.unet(en)
            dc = self.dc(unet)

            return dc

        elif source == "PseudoGT":
            for param in self.en_s.parameters():
                param.requires_grad = False
            for param in self.en_t.parameters():
                param.requires_grad = False
            for param in self.en_pgt.parameters():
                param.requires_grad = True
            for param in self.unet.parameters():
                param.requires_grad = True
            for param in self.dc.parameters():
                param.requires_grad = True

            en = self.en_pgt(x)
            unet = self.unet(en)
            dc = self.dc_raw(unet)

            return dc

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
