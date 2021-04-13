import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from loader import lowlightloader
from loader2 import lowlight
from network import T_enhancer,Discriminator
import utils
import config



torch.backends.cudnn.benchmark = True

def main():
    args = config.config()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"prefetch_factor":args.bs*8,"drop_last":True,"num_workers":8,"pin_memory":True}
    writer = SummaryWriter(f"/home/tejas/experimentations/image_enhancement/logs/Low_light_Tenhancer_{args.m}_LR_{args.lr}_EPOCHS_{args.epochs}_BS_{args.bs}/")

    print(device,kwargs)
    Adtpath = args.Adatapath
    Bdtpath = args.Bdatapath
    Adtspath = args.Adatapathts
    Bdtspath = args.Bdatapathts
    my_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=0.30),
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),inplace=True),
    ])


    my_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128)),
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),inplace=True),
    ])





    train_data = lowlight(Adtpath,Bdtpath,transforms=my_t)
    train_loader = DataLoader(train_data,batch_size=args.bs,shuffle=True,**kwargs)

    test_data = lowlight(Adtspath,Bdtspath,transforms=my_test)
    test_loader = DataLoader(test_data,batch_size=args.testbs,shuffle=False,**kwargs)


    Gen =T_enhancer(inchannels=3,f=6).to(device)
    Disc = Discriminator(inchannels=6,features=[64,128,256,512]).to(device)

    mae = nn.L1Loss().to(device)
    bce = nn.BCEWithLogitsLoss().to(device)
    mse = nn.MSELoss().to(device)

    me = {"mse":mse,"mae":mae}

    opt_gen = optim.Adam(Gen.parameters(),lr=args.lr,weight_decay=1e-5,betas=(0.5,0.999))
    opt_disc = optim.Adam(Disc.parameters(),lr=args.lr,weight_decay=1e-5,betas=(0.5,0.999))

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()



    if args.lm:
        utils.load_checkpoint(
            args.cpgen, Gen, opt_gen, args.lr,device
        )
        utils.load_checkpoint(
            args.cpdisc, Disc, opt_disc, args.lr,device
        )


    num_epochs = args.epochs
    print("STARTED Training----------------")
    Disc.train()
    Gen.train()
    for epochs in range(0,num_epochs+1):
        utils.train(train_loader,Disc,Gen,opt_disc,opt_gen,bce,me,g_scaler,d_scaler,device,writer,epochs,args)


        if args.save_model and epochs % 10 == 0:
            utils.save_checkpoint(Gen, opt_gen, filename=args.cpgen)
            utils.save_checkpoint(Disc, opt_disc, filename=args.cpdisc)
            utils.test(test_loader,Gen,Disc,device,epochs,args)

    print("-----------TRAINING ENDED")







    if args.save_model:
        torch.save(Gen.state_dict(),f"/home/tejas/experimentations/image_enhancement/model/{args.m}_Low_light_Tenhancer_GEN_BS_{args.bs}_LR_{args.lr}_EPOCHS_{args.epochs}.pt")
        torch.save(Disc.state_dict(),f"/home/tejas/experimentations/image_enhancement/model/{args.m}_Low_light_Tenhancer_DISC_BS_{args.bs}_LR_{args.lr}_EPOCHS_{args.epochs}.pt")
        print("model saved")


if __name__ == "__main__":
    main()
