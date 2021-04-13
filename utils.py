import torch
import cv2
from tqdm import tqdm
from torchvision.utils import save_image
import os
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import matplotlib.pylab as plt


def to_img(x,y,y_fake,ipath,tpath,gpath,epochs,idx,args):
    x = x*0.5 +0.5
    y = y*0.5 +0.5
    y_fake = y_fake*0.5 +0.5

    
    x = x.clamp(0,1)*255
    y = y.clamp(0,1)*255
    y_fake = y_fake.clamp(0,1)*255

    x = x[0].permute(1,2,0).detach().numpy()
    y = y[0].permute(1,2,0).detach().numpy()
    y_fake = y_fake[0].permute(1,2,0).detach().numpy()



    if not os.path.exists(args.resultspath):
        os.mkdir(args.resultspath)
    if not os.path.exists(args.resultspath+f"/{args.m}/"):
        os.mkdir(args.resultspath+f"/{args.m}/")
    if not os.path.exists(args.resultspath+f"/{args.m}/{epochs}/"):
        os.mkdir(args.resultspath+f"/{args.m}/{epochs}/")

    path=args.resultspath+f"/{args.m}/{epochs}/{idx}"


    # save_image(x,path+"_input.png",normalize=True)
    # save_image(y,path+"_target.png",normalize=True)
    # save_image(y_fake,path+"_generated.png",normalize=True)


    cv2.imwrite(path+"_input.png",x)
    cv2.imwrite(path+"_target.png",y)
    cv2.imwrite(path+"_generated.png",y_fake)
    
    ipath.append(path+"_input.png")
    tpath.append(path+"_target.png")
    gpath.append(path+"_generated.png")



    return (x,y,y_fake)


def plot_disc(D_real,D_fake,epochs,idx,args):

    D_real ,D_fake = D_real.cpu().data,D_fake.cpu().data

    D_real = D_real[0].permute(1,2,0).detach().numpy()
    D_fake = D_fake[0].permute(1,2,0).detach().numpy()


    if not os.path.exists(args.resultspath):
        os.mkdir(args.dresultspath)
    if not os.path.exists(args.dresultspath+f"/{args.m}/"):
        os.mkdir(args.dresultspath+f"/{args.m}/")
    if not os.path.exists(args.dresultspath+f"/{args.m}/{epochs}/"):
        os.mkdir(args.dresultspath+f"/{args.m}/{epochs}/")


    path=args.dresultspath+f"/{args.m}/{epochs}/{idx}"

    fig=plt.imshow(D_real,cmap='Blues')
    plt.axis("off")
    plt.savefig(path+"_real.png",bbox_inches='tight',pad_inches=0)
    fig=plt.imshow(D_fake,cmap='Reds')
    plt.axis("off")
    plt.savefig(path+"_fake.png",bbox_inches='tight',pad_inches=0)


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr,device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr



def train(train_loader,disc,gen,opt_disc,opt_gen,criterion_disc,criterion_gen,d_scaler,g_scaler,device,writer,epochs,args):
    loop = tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
    D=0
    G=0
    for idx ,(x1,x2,y) in loop:
        x1,x2,y = x1.to(device),x2.to(device),y.to(device)

        #Train Discriminator
        D_loss=0
        
        for _ in range(args.critic_itr):
            with torch.cuda.amp.autocast():
                y_fake = gen(x1,x2)
                D_real = disc(x1,y)
                D_fake = disc(x1,y_fake.detach())

                D_real_loss = criterion_disc(D_real,torch.ones_like(D_real))
                D_fake_loss = criterion_disc(D_fake,torch.zeros_like(D_fake))
                D_loss += (D_real_loss + D_fake_loss)
        
        D+=(D_loss/len(train_loader))

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()


        #Train Generator

        with torch.cuda.amp.autocast():
            D_fake = disc(x1,y_fake)
            G_fake_loss = criterion_disc(D_fake,torch.ones_like(D_fake))
            if args.mae:
                me = criterion_gen["mae"](y_fake,y)*args.lda

            if args.mse:
                me = criterion_gen["mse"](y_fake,y)*args.lda

            G_loss = G_fake_loss + me
        G+= (G_loss/len(train_loader))


        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        writer.add_scalar("Training loss Gen", G.item(), epochs * len(train_loader) +idx)
        writer.add_scalar("Training loss Disc", D.item(),epochs * len(train_loader) +idx)


        if idx % args.logs ==0:
            loop.set_description(f"[{epochs}/{args.epochs}]")
            loop.set_postfix(D_real = torch.sigmoid(D_real).mean().item(),
                             D_fake = torch.sigmoid(D_fake).mean().item(),
                             Disc_Loss = D.item(),
                             Gen_Loss = G.item())




def test(test_loader,gen,disc,device,epochs,args):
    print("Testing Models ---------")
    gen.eval()
    disc.eval()

    ssim_list =[]
    mse_list=[]
    ipath=[]
    tpath=[]
    gpath=[]
    dpath=[]

    print("Saving images and plots------")

    # with torch.cuda.amp.autocast():
    with torch.no_grad():

        for idx,(x1,x2,y) in enumerate(test_loader):
            x1,x2,y = x1.to(device),x2.to(device),y.to(device)
            y_fake = gen(x1,x2)
            D_real = disc(x1,y)
            D_fake = disc(x1,y_fake)

            x,y,y_fake = x1.cpu().data,y.cpu().data,y_fake.cpu().data

            x, y ,y_fake = to_img(x,y,y_fake,ipath,tpath,gpath,epochs,idx,args)
            plot_disc(D_real,D_fake,epochs,idx,args)
            mse_list.append(((y_fake-y)**2).mean())
            ssim_list.append(ssim(y,y_fake,data_range=y_fake.max()-y_fake.min(),multichannel=True))
            

    print("--------Done")

    print("Creating Excel sheet------")
    df = pd.DataFrame(data={"input_dir":ipath,"target_path":tpath,"gen_path":gpath,"ssim":ssim_list,"mse":mse_list})
    if not os.path.exists(args.resultcsv+f"/{args.m}/"):
        os.mkdir(args.resultcsv+f"/{args.m}/")
    df.to_csv(args.resultcsv+f"/{args.m}/Epoch_{str(epochs)}.csv")
    print("----Done")


    gen.train()
    disc.train()

    print("-------Done testing")
        




        
