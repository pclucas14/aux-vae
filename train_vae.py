import argparse
import time
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
from vaes.convHVAE_2level import VAE
# from vaes.PixelHVAE_2level import VAE
# from utils.optimizer import AdamNormGrad
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

# cuda
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
# random seed
parser.add_argument('--seed', type=int, default=14, metavar='S',
                    help='random seed (default: 14)')
# model: latent size, input_size, so on
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

# model: model name, prior
parser.add_argument('--model_name', type=str, default='vae', metavar='MN',
                    help='model name: vae, hvae_2level, convhvae_2level, pixelhvae_2level')

parser.add_argument('--prior', type=str, default='vampprior', metavar='P',
                    help='prior: standard, vampprior')

parser.add_argument('--input_type', type=str, default='continuous', metavar='IT',
                    help='type of the input: binary, gray, continuous')

# experiment
parser.add_argument('--S', type=int, default=5000, metavar='SLL',
                    help='number of samples used for approximating log-likelihood')
parser.add_argument('--MB', type=int, default=100, metavar='MBLL',
                    help='size of a mini-batch used for approximating log-likelihood')

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', metavar='DN',
                    help='name of the dataset: static_mnist, dynamic_mnist, omniglot, caltech101silhouettes, histopathologyGray, freyfaces, cifar10')

parser.add_argument('--dynamic_binarization', action='store_true', default=False,
                    help='allow dynamic binarization')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

sample_batch_size = 25
obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : x#(x - .5) * 2.
rescaling_inv = lambda x : x#.5 * x  + .5

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
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))

model = VAE(args)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

print('starting training')
for epoch in range(args.epochs):
    model.train(True)
    torch.cuda.synchronize()
    train_loss, train_re, train_kl = 0., 0., 0.
    time_ = time.time()
    model.train()

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
        loss, RE, KL, _ = model.calculate_loss(input, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        train_re   += -RE.data[0]
        train_kl   += KL.data[0]
        if batch_idx % 50 == 49 : 
            deno = 50*args.batch_size*np.prod(obs)*np.log(2.)
            print('loss : {:.4f}, kl : {:.4f}, elbo : {:.4f}, time : {:.4f}'.format(
                (train_loss / deno), 
                (train_kl / deno),
                (train_re + train_kl) / deno,
                (time.time() - time_)))
            train_loss, train_re, train_kl = 0., 0., 0.
            time_ = time.time()
    
    # decrease learning rate
    scheduler.step()
    
    torch.cuda.synchronize()
    model.eval()
    test_loss = 0.
    
    print('test time!')
    for batch_idx, (input,_) in enumerate(test_loader):
        input = input.cuda(async=True)
        input = Variable(input)
        loss, RE, KL, _ = model.calculate_loss(input, beta)
        test_loss += loss.data[0]
    print('test loss : %s' % (test_loss / (batch_idx*args.batch_size*np.log(2.)*np.prod(obs))))
   
    print('sampling...')
    sample_t = model.reconstruct_x(input).data #sample(model)
    sample_t = rescaling_inv(sample_t)
    utils.save_image(sample_t,'snapshots/vae_{}_{}.png'.format(args.dataset, epoch), nrow=5)
    real_img = rescaling_inv(input.data)
    utils.save_image(real_img,'snapshots/vae_real_{}_{}.png'.format(args.dataset, epoch), nrow=5)
    
    if (epoch + 1) % args.save_interval == 0: 
        torch.save(model.state_dict(), 'models/vae_{}_{}.pth'.format(args.dataset, epoch))
