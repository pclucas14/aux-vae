import argparse
import time
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
from vaes.convHVAE_2level import VAE
from utils.utils import * 
from model import * 
# from utils.optimizer_vae import AdamNormGrad
import numpy as np
import pdb
import os
import datetime
from utils.load_data import load_dataset

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # # # # # # # # # #
# START EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #


# Training settings
parser = argparse.ArgumentParser(description='VAE+VampPrior')
# arguments for optimization
parser.add_argument('-t', '--save_interval', type=int, default=10,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-i', '--data_dir', type=str,
                    default='datasets', help='Location for the dataset')
parser.add_argument('--batch_size', type=int, default=100, metavar='BStrain',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='BStest',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=2000, metavar='E',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('--early_stopping_epochs', type=int, default=50, metavar='ES',
                    help='number of epochs for early stopping')

parser.add_argument('--warmup', type=int, default=100, metavar='WU',
                    help='number of epochs for warmu-up')
parser.add_argument('--load_params', action='store_true', default=False,
                    help='load parameters from file')

# cuda
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
# random seed
parser.add_argument('--seed', type=int, default=14, metavar='S',
                    help='random seed (default: 14)')
# vae: latent size, input_size, so on
parser.add_argument('--z1_size', type=int, default=40, metavar='M1',
                    help='latent size')
parser.add_argument('--z2_size', type=int, default=40, metavar='M2',
                    help='latent size')
parser.add_argument('--input_size', type=int, default=[3, 32, 32], metavar='D',
                    help='input size')

parser.add_argument('--activation', type=str, default=None, metavar='ACT',
                    help='activation function')

parser.add_argument('--number_components', type=int, default=500, metavar='NC',
                    help='number of pseudo-inputs')
parser.add_argument('--pseudoinputs_mean', type=float, default=-0.05, metavar='PM',
                    help='mean for init pseudo-inputs')
parser.add_argument('--pseudoinputs_std', type=float, default=0.01, metavar='PS',
                    help='std for init pseudo-inputs')

parser.add_argument('--use_training_data_init', action='store_true', default=False,
                    help='initialize pseudo-inputs with randomly chosen training data')

# vae: model name, prior
parser.add_argument('--model_name', type=str, default='vae', metavar='MN',
                    help='model name: vae, hvae_2level, convhvae_2level, pixelhvae_2level')

parser.add_argument('--prior', type=str, default='vampprior', metavar='P',
                    help='prior: standard, vampprior')

parser.add_argument('--input_type', type=str, default='continuous', metavar='IT',
                    help='type of the input: binary, gray, continuous')

# pixel cnn
parser.add_argument('-q', '--nr_resnet', type=int, default=3,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
# experiment
parser.add_argument('--S', type=int, default=5000, metavar='SLL',
                    help='number of samples used for approximating log-likelihood')
parser.add_argument('--MB', type=int, default=100, metavar='MBLL',
                    help='size of a mini-batch used for approximating log-likelihood')

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', metavar='DN',
                    help='name of the dataset: mnist, cifar10')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

model_name = 'AGAVE_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(args.lr, args.nr_resnet, args.nr_filters)
sample_batch_size = 25
obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5

kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if 'mnist' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                        train=True, transform=ds_transforms), batch_size=args.batch_size, 
                            shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
elif 'cifar' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))

vae = VAE(args)
vae = vae.cuda()
pcnn = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
pcnn = pcnn.cuda()

if args.load_params : 
    pcnn.load_state_dict(torch.load('models/AGAVE_pcnn_cifar10_0.pth'))
    vae.load_state_dict(torch.load('models/AGAVE_vae_cifar10_0.pth'))
    print('model parameters loaded')

optimizer_vae  = optim.Adam(vae.parameters(), lr=args.lr)
optimizer_pcnn = optim.Adam(pcnn.parameters(), lr=args.lr)
scheduler_vae  = lr_scheduler.StepLR(optimizer_vae, step_size=1, gamma=args.lr_decay)
scheduler_pcnn = lr_scheduler.StepLR(optimizer_pcnn, step_size=1, gamma=args.lr_decay)

def sample(pcnn, vae_out):
    data = torch.zeros(vae_out.size(0), obs[0], obs[1], obs[2])
    data = data.cuda()
    for i in range(obs[1]):
        for j in range(obs[2]):
            data_v = Variable(data, volatile=True)
            out   = pcnn(data_v, vae_out, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

print('starting training')
for epoch in range(args.epochs):
    vae.train(True)
    pcnn.train(True)
    torch.cuda.synchronize()
    train_loss, train_re, train_kl, train_pcnn = 0., 0., 0., 0.
    time_ = time.time()
    vae.train()

    if args.warmup == 0:
        beta = 1.
    else:
        beta = 1.* epoch / args.warmup
        if beta > 1.:
            beta = 1.
    
    print('beta: {}'.format(beta))
    for batch_idx, (input,_) in enumerate(train_loader):
        input = input.cuda(async=True)
        input = Variable(input)
        loss, RE, KL, vae_out = vae.calculate_loss(rescaling_inv(input), beta)
        optimizer_vae.zero_grad()
        loss.backward()
        optimizer_vae.step()
        vae_out = vae_out.detach()
        output = pcnn(input, rescaling(vae_out))
        loss_pcnn = loss_op(input, output)
        optimizer_pcnn.zero_grad()
        loss_pcnn.backward()
        optimizer_pcnn.step()
        train_loss += loss.data[0]
        train_re   += -RE.data[0]
        train_kl   += KL.data[0]
        train_pcnn += loss_pcnn.data[0]

        if batch_idx % 50 == 49 : 
            deno = 50 * args.batch_size * np.prod(obs) * np.log(2.)
            print('loss : {:.4f}, kl : {:.4f}, elbo_vae : {:.4f}, recon_pcnn : {:.4f}, time : {:.4f}'.format(
                (train_loss / deno), 
                (train_kl / deno),
                (train_re + train_kl) / deno,
                (train_pcnn / deno),
                (time.time() - time_)))
            train_loss, train_re, train_kl, train_pcnn = 0., 0., 0., 0.
            time_ = time.time()
    
    # decrease learning rate
    scheduler_vae.step()
    scheduler_pcnn.step()
    
    torch.cuda.synchronize()
    vae.eval()
    pcnn.eval()
    test_loss, test_re, test_kl, test_pcnn = 0., 0., 0., 0.
    
    print('test time!')
    for batch_idx, (input,_) in enumerate(test_loader):
        input = input.cuda(async=True)
        input_var = Variable(input)
        loss, RE, KL, vae_out = vae.calculate_loss(rescaling_inv(input_var), beta)
        output = pcnn(input_var, rescaling(vae_out))
        loss_pcnn = loss_op(input_var, output)
        test_loss += loss.data[0]
        test_re   += -RE.data[0]
        test_kl   += KL.data[0]
        test_pcnn += loss_pcnn.data[0]
        del output, vae_out, loss, loss_pcnn, RE, KL
    
    deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
    print('test loss : {:.4f}, kl : {:.4f}, elbo_vae : {:.4f}, recon_pcnn : {:.4f}'.format(
                    (test_loss / deno), 
                    (test_kl / deno),
                    (test_re + train_kl) / deno,
                    (test_pcnn / deno)))
   
    print('sampling...')
    sample_t = vae.reconstruct_x(rescaling_inv(input_var))
    utils.save_image(sample_t.data,'snapshots/{}_vae_{}.png'.format(model_name, epoch), nrow=5)
    real_img = rescaling_inv(input_var.data)
    utils.save_image(real_img,'snapshots/{}_real_{}.png'.format(model_name, epoch), nrow=5)
    
    if (epoch + 1) % args.save_interval == 0: 
        torch.save(vae.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
        torch.save(pcnn.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
        
        sample_t = sample(pcnn, rescaling_inv(sample_t))
        utils.save_image(sample_t,'snapshots/{}_{}.png'.format(model_name, epoch), nrow=5)
