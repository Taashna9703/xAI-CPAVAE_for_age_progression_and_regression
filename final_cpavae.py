from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pickle
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch
'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find("Linear") !=-1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
'''
def one_hot(labelTensor,batchSize,n_l,use_cuda=False):
    labelTensor = labelTensor.long()
    oneHot = - torch.ones(batchSize*n_l).view(batchSize,n_l)
    for i,j in enumerate(labelTensor):
        oneHot[i,j] = 1
    if use_cuda:
        return Variable(oneHot).cuda()
    else:
        return Variable(oneHot)

def TV_LOSS(imgTensor,img_size=128):
    x = (imgTensor[:,:,1:,:]-imgTensor[:,:,:img_size-1,:])**2
    y = (imgTensor[:,:,:,1:]-imgTensor[:,:,:,:img_size-1])**2

    out = (x.mean(dim=2)+y.mean(dim=3)).mean()
    return out

layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
default_content_layers = ['relu1_1', 'relu2_1', 'relu3_1']

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='/media/dllab-1/My_disk2/Praveen/EXPI/CPAVAE/MRCD_Asian_Black_White_Dataset/PR_AsianChildData',
                    help='path to dataset folder (must follow PyTorch ImageFolder structure)')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers, default=2', default=2)
parser.add_argument('--batch_size', type=int,
                    default=64, help='input batch size, default=128')
parser.add_argument('--image_size', type=int, default=128,
                    help='height/width length of the input images, default=64')
parser.add_argument('--nz', type=int, default=50,
                    help='size of the latent vector z, default=100')
parser.add_argument('--nef', type=int, default=64,
                    help='number of output channels for the first encoder layer, default=32')
parser.add_argument('--ndf', type=int, default=64,
                    help='number of output channels for the first decoder layer, default=32')
parser.add_argument('--instance_norm', action='store_true',
                    help='use instance norm layer instead of batch norm')
parser.add_argument('--content_layers', type=str, nargs='?', default=None,
                    help='name of the layers to be used to compute the feature perceptual loss, default=[relu3_1, relu4_1, relu5_1]')
parser.add_argument('--niter', type=int, default=51,
                    help='number of epochs to train for, default=10')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate, default=0.0005')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam, default=0.5')
parser.add_argument('--cuda', action='store_true',default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
'''
parser.add_argument('--encoder', default='/home/mnit/PycharmProjects/DFCVA/Modified_DFCAE_CAAE_VGG123/outputVGGISCD/encoder_epoch_30.pth',
                    help="path to encoder (to continue training)")
parser.add_argument('--decoder', default='/home/mnit/PycharmProjects/DFCVA/Modified_DFCAE_CAAE_VGG123/outputVGGISCD/decoder_epoch_30.pth',
                    help="path to decoder (to continue training)")
'''
parser.add_argument('--encoder', default='',
                    help="path to encoder (to continue training)")
parser.add_argument('--decoder', default='',
                    help="path to decoder (to continue training)")
parser.add_argument('--outf', default='./outputVGGUTKFaceCat4Epoch111',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--log_interval', type=int, default=1, help='number of iterations between each stdout logging, default=1')
parser.add_argument('--img_interval', type=int, default=100, help='number of iterations between each image saving, default=100')


args = parser.parse_args()
print(args)
## boolean variable indicating whether cuda is available
use_cuda = torch.cuda.is_available()
#n_channel = 3
#img_size = 128
#batchSize = 20
#n_encode = 64`1

n_z = 50
#n_l =10
n_l =4

n_channel = 3
n_disc = 16
n_gen = 64

use_cuda = torch.cuda.is_available()
n_age = int(n_z/n_l) #12
n_gender = int(n_z/2) #25
print("USE_CUDA=", use_cuda)
try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manual_seed is None:
    args.manual_seed = random.randint(1, 10000)
print("Random Seed: ", args.manual_seed)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manual_seed)

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Dataset loading
# Normalization mean and standard deviation are set accordingly to the ones used
# to train the vgg19 in torchvision model zoo
# https://github.com/pytorch/vision
transform = transforms.Compose([
    transforms.Resize(args.image_size),
    #transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))])
datafolder = dset.ImageFolder(root=args.dataroot, transform=transform)
dataloader = torch.utils.data.DataLoader(
datafolder, shuffle=True, batch_size=args.batch_size, num_workers=args.workers, drop_last=True)

ngpu = int(args.ngpu)
nz = int(args.nz)
nef = int(args.nef)
ndf = int(args.ndf)
nc = 3
out_size = args.image_size // 16  # 64
if args.instance_norm:
    Normalize = nn.InstanceNorm2d
else:
    Normalize = nn.BatchNorm2d
if args.content_layers is None:
    content_layers = default_content_layers
else:
    content_layers = args.content_layers

if use_cuda:
    BCE = nn.BCELoss().cuda()
    L1  = nn.L1Loss().cuda()
    CE = nn.CrossEntropyLoss().cuda()
def weights_init(m):
    '''
    Custom weights initialization called on encoder and decoder.
    '''
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight.data, a=0.01)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, std=0.015)
        m.bias.data.zero_()


class _VGG(nn.Module):
    '''
    Classic pre-trained VGG19 model.
    Its forward call returns a list of the activations from
    the predefined content layers.
    '''

    def __init__(self, ngpu):
        super(_VGG, self).__init__()

        self.ngpu = ngpu
        features = models.vgg19(pretrained=True).features

        self.features = nn.Sequential()
        for i, module in enumerate(features):
            name = layer_names[i]
            self.features.add_module(name, module)

    def forward(self, input):
        batch_size = input.size(0)
        all_outputs = []
        output = input
        for name, module in self.features.named_children():
            # print("VGG without Parallel...")
            output = module(output)
            if name in content_layers:
                all_outputs.append(output.view(batch_size, -1))
        return all_outputs


descriptor = _VGG(ngpu)
print(descriptor)


class _Encoder(nn.Module):
    '''
    Encoder module, as described in :
    https://arxiv.org/abs/1610.00291
    '''

    def __init__(self, ngpu):
        super(_Encoder, self).__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.ReLU(True),
            #nn.LeakyReLU(0.2, True),
            #Normalize(nef),

            nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            nn.ReLU(True),
            #nn.LeakyReLU(0.2, True),
            #Normalize(nef * 2),

            nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            nn.ReLU(True),
            #nn.LeakyReLU(0.2, True),
           # Normalize(nef * 4),

            nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            nn.ReLU(True),
            #nn.LeakyReLU(0.2, True),
            #Normalize(nef * 8)
        )
        self.mean = nn.Linear(nef * 8 * out_size * out_size, nz)
        self.logvar = nn.Linear(nef * 8 * out_size * out_size, nz)

    def sampler(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, input):
        batch_size = input.size(0)
        #print("Decoder without Parallel...")
        hidden = self.encoder(input)
        hidden = hidden.view(batch_size, -1)
        mean, logvar = self.mean(hidden), self.logvar(hidden)
        latent_z = self.sampler(mean, logvar)
        return latent_z


encoder = _Encoder(ngpu)
encoder.apply(weights_init)
if args.encoder != '':
    encoder.load_state_dict(torch.load(args.encoder))
print(encoder)


class _Decoder(nn.Module):
    '''
    Decoder module, as described in :
    https://arxiv.org/abs/1610.00291
    '''

    def __init__(self, ngpu):
        super(_Decoder, self).__init__()
        self.ngpu = ngpu
        self.decoder_dense = nn.Sequential(
            nn.Linear(n_z+n_l*n_age+n_gender, ndf * 8 * out_size * out_size),
            nn.LeakyReLU(0.2, True),
        )
        self.decoder_conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 8, ndf * 4, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(ndf * 4, 1e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 4, ndf * 2, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(ndf * 2, 1e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 2, ndf, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(ndf, 1e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf, nc, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z,age,gender):
        batch_size = z.size(0)
        #print("Latent Vector Size",z.size())
        ## duplicate age & gender conditions as descripted in https://github.com/ZZUTK/Face-Aging-CAAE
        l = age.repeat(1, n_age)  # size = 20 * 48
        #print("Age repeat Size", l.size())
        k = gender.view(-1, 1).repeat(1, n_gender)  # size = 20 * 25
        #print("Gender ", k.size())
        x = torch.cat([z, l, k.float()], dim=1)  # size = 20 * 123
        #print("After Concat x",x.size())
       #print("Decoder without Parallel...")
        #print('Dcoder input ',batch_size,ndf*16,out_size)
        hidden = self.decoder_dense(x).view(batch_size,ndf * 8, out_size, out_size)
        output = self.decoder_conv(hidden)
        return output

class Dz(nn.Module):
    def __init__(self):
        super(Dz,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_z,n_disc*4),
            nn.ReLU(),

            nn.Linear(n_disc*4,n_disc*2),
            nn.ReLU(),

            nn.Linear(n_disc*2,n_disc),
            nn.ReLU(),

            nn.Linear(n_disc,1),
            nn.Sigmoid()
        )

    # z size = 20 * 50
    def forward(self,z):
        return self.model(z) # size = 20 * 1

class Dimg(nn.Module):
    def __init__(self):
        super(Dimg,self).__init__()
        self.conv_img = nn.Sequential(
            nn.Conv2d(n_channel,n_disc,4,2,1),
        )
        self.conv_l = nn.Sequential(
            nn.ConvTranspose2d(n_l*n_age+n_gender, n_l*n_age+n_gender, 64, 1, 0),
            nn.ReLU()
        )
        self.total_conv = nn.Sequential(
            nn.Conv2d(n_disc+n_l*n_age+n_gender,n_disc*2,4,2,1),
            nn.ReLU(),

            nn.Conv2d(n_disc*2,n_disc*4,4,2,1),
            nn.ReLU(),

            nn.Conv2d(n_disc*4,n_disc*8,4,2,1),
            nn.ReLU()
        )

        self.fc_common = nn.Sequential(
            nn.Linear(8*8*args.image_size,1024),
            nn.ReLU()
        )
        self.fc_head1 = nn.Sequential(
            nn.Linear(1024,1),
            nn.Sigmoid()
        )
        self.fc_head2 = nn.Sequential(
            nn.Linear(1024,n_l),
            nn.Softmax()
        )

    # format = noOfImages * noOfChannel * imageHeight * imageWidth
    # img size = 20 * 3 * 128 * 128
    # age size = 20 * 4 * 1 * 1
    # gender size = 20 * 1 * 1 * 1
    def forward(self,img,age,gender):
        ## duplicate age & gender conditions as descripted in https://github.com/ZZUTK/Face-Aging-CAAE
        l = age.repeat(1,n_age,1,1,) # size = 20 * 48 * 1 * 1
        k = gender.repeat(1,n_gender,1,1,) # size = 20 * 25 * 1 * 1
        conv_img = self.conv_img(img) # size = 20 * 16 * 64 * 64
        conv_l   = self.conv_l(torch.cat([l,k],dim=1)) # torch.cat([l,k] size = 20 * 73 * 1 * 1, # size = 20 * 73 * 64 * 64
        catted   = torch.cat((conv_img,conv_l),dim=1) # size = 20 * 89 * 64 * 64
        total_conv = self.total_conv(catted).view(-1,8*8*args.image_size) # size = 20 * 128 * 8 * 8
        body = self.fc_common(total_conv) # size = 20  * 1024

        head1 = self.fc_head1(body) # size = 20 * 1
        head2 = self.fc_head2(body) # size = 20 * 4

        return head1,head2


decoder = _Decoder(ngpu)
decoder.apply(weights_init)
if args.decoder != '':
    decoder.load_state_dict(torch.load(args.decoder))
print(decoder)
netD_z  = Dz().cuda()
netD_z.apply(weights_init)
netD_img = Dimg().cuda()
netD_img.apply(weights_init)

mse = nn.MSELoss()
def fpl_criterion(recon_features, targets):
    fpl = 0
    for f, target in zip(recon_features, targets):
        fpl += mse(f, target.detach())#.div(f.size(1))
    return fpl

kld_criterion = nn.KLDivLoss()


input = torch.FloatTensor(
    args.batch_size, nc, args.image_size, args.image_size)
latent_labels = torch.FloatTensor(args.batch_size, nz).fill_(1)

if args.cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    descriptor = descriptor.cuda()
    input = input.cuda()
    latent_labels = latent_labels.cuda()

input = Variable(input)
latent_labels = Variable(latent_labels)

# setup optimizer
#parameters = list(encoder.parameters()) + list(decoder.parameters())+ list(netD_z.parameters()) + list(netD_img.parameters())
#optimizer = optim.Adam(parameters, lr=args.lr, betas=(args.beta1, 0.999))
## build optimizer for each networks
optimizerE = optim.Adam(encoder.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerD_z = optim.Adam(netD_z.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerD_img = optim.Adam(netD_img.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerD = optim.Adam(decoder.parameters(),lr=0.0002,betas=(0.5,0.999))
## fixed variables to regress / progress age
fixed_l = -torch.ones(32*4).view(32,4)
for i,l in enumerate(fixed_l):
    l[i//8] = 1

fixed_l_v = Variable(fixed_l)

if use_cuda:
    fixed_l_v = fixed_l_v.cuda()

encoder.train()
decoder.train()

train_loss = 0
d_loss = []
g_loss = []
for epoch in range(args.niter):
    for i, (img_data, img_label) in enumerate(dataloader):
        # make image variable and class variable
        torch.cuda.empty_cache()
        img_data_v = Variable(img_data)
        img_age = img_label / 2  # size = no of image * 1
        img_gender = img_label % 2 * 2 - 1  # size = no of image * 1

        img_age_v = Variable(img_age).view(-1, 1)
        img_gender_v = Variable(img_gender.float())

        if epoch == 0 and i == 0:
            fixed_noise = img_data[:8].repeat(4, 1, 1, 1)  # size = 32 * 3 * 128 * 128
            fixed_g = img_gender[:8].view(-1, 1).repeat(4, 1)  # size = 32 * 1
            fixed_img_v = Variable(fixed_noise)
            fixed_g_v = Variable(fixed_g)
            pickle.dump(fixed_noise, open("fixed_noise.p", "wb"))

            if use_cuda:
                fixed_img_v = fixed_img_v.cuda()
                fixed_g_v = fixed_g_v.cuda()
                vutils.save_image(fixed_img_v.data,
                                  '{}/initial_inputs.png'.format(args.outf),
                                  normalize=True)
        if use_cuda:
            img_data_v = img_data_v.cuda()
            img_age_v = img_age_v.cuda()
            img_gender_v = img_gender_v.cuda()

        # make one hot encoding version of label
        batchSize = img_data_v.size(0)
        age_ohe = one_hot(img_age, batchSize, n_l, use_cuda)  # size = noOfImages * n_l

        # prior distribution z_star, real_label, fake_label
        z_star = Variable(torch.FloatTensor(batchSize * n_z).uniform_(-1, 1)).view(batchSize, n_z)
        real_label = Variable(torch.ones(batchSize).fill_(1)).view(-1, 1)
        fake_label = Variable(torch.ones(batchSize).fill_(0)).view(-1, 1)

        if use_cuda:
            z_star, real_label, fake_label = z_star.cuda(), real_label.cuda(), fake_label.cuda()

        ## train Encoder and Generator with reconstruction loss

        optimizerE.zero_grad()
        optimizerD.zero_grad()
        input.data.copy_(img_data)

        latent_z = encoder(input)
        targets = descriptor(input)
        kld = kld_criterion(F.log_softmax(latent_z), latent_labels)
        kld.backward(create_graph=True)

        recon = decoder(latent_z,age_ohe,img_gender_v)
        recon_features = descriptor(recon)
        fpl = fpl_criterion(recon_features, targets)
        #fpl.backward()
        ## EG_loss 3. GAN loss - z
        # Dz_prior = netD_z(z_star)
        Dz = netD_z(latent_z)
        # send z to d_z
        Ez_loss = BCE(Dz, real_label)
        loss = kld + fpl
        train_loss += loss.item()
        # EG_loss 2. GAN loss - image
        D_reconst, _ = netD_img(recon, age_ohe.view(batchSize, n_l, 1, 1), img_gender_v.view(batchSize, 1, 1, 1))
        # output of G+l sent to d_img
        G_img_loss = BCE(D_reconst, real_label)

        ## EG_loss 4. TV loss - G
        reconst = decoder(latent_z.detach(), age_ohe, img_gender_v)
        # Total Variance Loss in G
        G_tv_loss = TV_LOSS(reconst)

        EG_loss = loss + 0.0001 * G_img_loss + 0.01 * Ez_loss + G_tv_loss
        EG_loss.backward()
        optimizerE.step()
        optimizerD.step()
        ## train netD_z with prior distribution U(-1,1)
        netD_z.zero_grad()
        Dz_prior = netD_z(z_star)
        Dz = netD_z(latent_z.detach())
        # Total Discriminator_z Loss (pass z to d_z and pass z_prior in d_z)
        Dz_loss = BCE(Dz_prior, real_label) + BCE(Dz, fake_label)
        Dz_loss.backward()
        optimizerD_z.step()
        ## train D_img with real images
        netD_img.zero_grad()
        D_img, D_clf = netD_img(img_data_v, age_ohe.view(batchSize, n_l, 1, 1), img_gender_v.view(batchSize, 1, 1, 1))
        D_reconst, _ = netD_img(reconst.detach(), age_ohe.view(batchSize, n_l, 1, 1),
                                img_gender_v.view(batchSize, 1, 1, 1))

        # Total Discriminator_img Loss (pass x+l to d_img and pass G+l in d_img)
        D_loss = BCE(D_img, real_label) + BCE(D_reconst, fake_label)
        D_loss.backward()

        optimizerD_img.step()
        #optimizerD_z.step()
        '''
        if i % args.log_interval == 0:
            print('[{}/{}][{}/{}] FPL: {:.4f} KLD: {:.4f}'.format(
            epoch, args.niter, i, len(dataloader),
                  fpl.data[0], kld.data[0]))
        '''
       # d_loss.append(D_loss)
       # g_loss.append(EG_loss)
        if i % args.img_interval == 0:
            vutils.save_image(input.data,
                              '{}/inputs.png'.format(args.outf),
                              normalize=True)
          #  vutils.save_image(recon.data,
          #                    '{}/reconstructions_epoch_{:03d}.png'.format(
          #                        args.outf, epoch),
          #                    normalize=True)

    ## save fixed img for every 20 step
    #print("========save==============")
    fixed_z = encoder(fixed_img_v)
    #print("Latent Vector fixed_z ",fixed_z.size())
    #print("Latent Vector fixed_l_v ",fixed_l_v.size())
    #print("Latent Vector fixed_g_v ",fixed_g_v.size())
    fixed_fake = decoder(fixed_z, fixed_l_v, fixed_g_v)
    vutils.save_image(fixed_fake.data,
                      '%s/reconst_epoch%03d.png' % (args.outf, epoch + 1),
                      normalize=True)

    # do checkpointing
    if epoch % 5 == 0:
        torch.save(encoder.state_dict(), '{}/encoder_epoch_{}.pth'.format(args.outf, epoch))
        torch.save(decoder.state_dict(), '{}/decoder_epoch_{}.pth'.format(args.outf, epoch))
        torch.save(netD_z.state_dict(), '{}/dz_epoch_{}.pth'.format(args.outf, epoch))
        torch.save(netD_img.state_dict(), '{}/dimag_epoch_{}.pth'.format(args.outf, epoch))
    msg1 = "epoch:{}, step:{}".format(epoch + 1, i + 1)
    msg2 = format("FPL loss :%f" % (fpl.item()), "<30") + "|" + format("KLD :%f" % (kld.item()), "<30")
    msg3 = format("G_img_loss:%f" % (G_img_loss.item()), "<30")
    msg4 = format("G_tv_loss:%f" % (G_tv_loss.item()), "<30") + "|" + "Ez_loss:%f" % (Ez_loss.item())
    msg5 = format("D_img:%f" % (D_img.mean().item()), "<30") + "|" + format(
        "D_reconst:%f" % (D_reconst.mean().item()), "<30") \
           + "|" + format("D_loss:%f" % (D_loss.item()), "<30")
    msg6 = format("D_z:%f" % (Dz.mean().item()), "<30") + "|" + format("D_z_prior:%f" % (Dz_prior.mean().item()),
                                                                        "<30") \
           + "|" + format("Dz_loss:%f" % (Dz_loss.item()), "<30")

    print()
    print(msg1)
    print(msg2)
    print(msg3)
    print(msg4)
    print(msg5)
    print(msg6)
    print()
    print("-" * 80)
# save a plot of the costs
'''
plt.clf()
plt.plot(d_loss, label='discriminator cost')
plt.plot(g_loss, label='generator cost')
plt.legend()
plt.savefig('cost_vs_iteration.png')
'''
