import torch.nn as nn
import torch
from torchvision.models import vgg16
from src import Histformer
from src import PDFLabformer
from src import sRGBformer

class HistNet(nn.Module):
    def __init__(self, inchnls=9, em_dim=16, device='cuda', wbset_num=3):

        self.inchnls = inchnls
        self.device = device
        super(HistNet, self).__init__()
        self.net = Histformer.Hist_Histoformer(in_chans=inchnls, embed_dim=em_dim, token_projection='linear', token_mlp='TwoDCFF', wbset_num=wbset_num).cuda().to(self.device)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """ Forward function"""
        # print('x_hist', x.shape)
        out_hist, hist_W = self.net(x)
        # print('out_hist', out_hist.shape)
        return out_hist, hist_W


class PDFLabNet(nn.Module):
    def __init__(self, inchnls=9, em_dim=16, device='cuda', wbset_num=3):

        self.inchnls = inchnls
        self.device = device
        super(PDFLabNet, self).__init__()
        self.net = PDFLabformer.Lab_Histoformer(in_chans=inchnls, embed_dim=em_dim, token_projection='linear', token_mlp='TwoDCFF', wbset_num=wbset_num).cuda().to(self.device)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """ Forward function"""
        # print('x_lab', x.shape)
        out_PDFLab, PDFLab_W = self.net(x)
        # print('out_PDFLab', out_PDFLab.shape)
        return out_PDFLab, PDFLab_W


class sRGBNet(nn.Module):
    def __init__(self, inchnls=9, em_dim=16, device='cuda', wbset_num=3):
        """ Network constructor. """
        self.outchnls = int(inchnls/3)
        self.inchnls = inchnls
        self.device = device
        super(sRGBNet, self).__init__()
        self.wbset_num = wbset_num
        self.net = sRGBformer.CAFormer(embed_dim=em_dim, in_chans=inchnls, token_projection='linear', wbset_num=wbset_num).cuda().to(self.device)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hist_W, PDFLab_W):
        """ Forward function"""
        # print('x_srgb',x.shape)#x torch.Size([1, 9, 128, 128])
        weights = self.net(x, hist_W, PDFLab_W)
        weights = torch.clamp(weights, -1000, 1000)
        # print('weights',weights.shape)# torch.Size([1, 3, 128, 128])
        weights = self.softmax(weights)
        out_img = torch.unsqueeze(weights[:, 0, :, :], dim=1) * x[:, :3, :, :]

        for i in range(1, int(self.wbset_num)):
            out_img += torch.unsqueeze(weights[:, i, :, :],dim=1) * x[:, (i * 3):3 + (i * 3), :, :]
        return out_img, weights


# if __name__ == '__main__':
#     x = torch.rand(4, 9, 64).cuda()
#     pdfx = torch.rand(4, 9, 64, 64).cuda()
#     net_h = HistNet(9, 16)
#     net_pl = PDFLabNet(9, 16)
#     nets = sRGBNet(9, 16)
#     hist_result, hist_W = net_h(x)
#     PDFLab_result, PDFlab_W  = net_pl(x)
#     y, w = nets(pdfx, hist_W, PDFlab_W)

