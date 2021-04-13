import argparse

def config():
    parser = argparse.ArgumentParser(description="Depth based FLASH AI")

    parser.add_argument(
        "--bs",
        type=int,
        default=16,
        metavar="N",
        help="batch size for training (default 16)",
    )

    parser.add_argument(
        "--testbs",
        type=int,
        default=1,
        metavar="N",
        help="test batch size (default 1)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="train epochs (default 100)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        metavar="LR",
        help="learning rate (default 0.001)",
    )

    parser.add_argument(
        "--no-cuda",action="store_true",default=False,help="disables CUDA training",
    )

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--logs",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )


    parser.add_argument(
        "--resultspath",
        type=str,
        default="/home/tejas/experimentations/image_enhancement/results/Gen/",
        help="path to store test images epoch wise (default not set)",
    )

    parser.add_argument(
        "--dresultspath",
        type=str,
        default="/home/tejas/experimentations/image_enhancement/results/Disc/",
        help="path to store test discriminator images epoch wise (default not set)",
    )

    parser.add_argument(
        "--resultcsv",
        type=str,
        default="/home/tejas/experimentations/image_enhancement/results/CSV/",
        help="path to store csv with evaluation metric (default not set)",
    )

    parser.add_argument(
        "--Adatapath",
        type=str,
        default="/home/tejas/Desktop/low_light/images/",
        help="path for image dataset with flash and ambient images (default not set)",
    )
    parser.add_argument(
        "--Bdatapath",
        type=str,
        default="/media/tejas/TAS/arch_data/images/",
        help="path for image dataset with flash and ambient images (default not set)",
    )

    parser.add_argument(
        "--Adatapathts",
        type=str,
        default="/home/tejas/Desktop/low_light/validation/",
        help="path for image dataset for testing with flash and ambient images (default not set)",
    )

    parser.add_argument(
        "--Bdatapathts",
        type=str,
        default="/media/tejas/TAS/arch_data/validation/",
        help="path for image dataset for testing with flash and ambient images (default not set)",
    )

    parser.add_argument(
        "--m",
        type=int,
        default=1,
        help="m is experimentation count to save models and summary (default 1)",
    )

    parser.add_argument(
        "--lda",
        type=int,
        default=100,
        help="lda lambda multipled with MAE loss as penality (default 100)",
    )

    parser.add_argument(
        "--critic-itr",
        type=int,
        default=1,
        help=" critic -itr is no of times dicriminator trained in each step (default 5)",
    )

    parser.add_argument(
        "--lm",
        type=bool,
        default=False,
        help="lm is load model from checkpoint (default False)",
    )


    parser.add_argument(
        "--mse",
        type=bool,
        default=True,
        help="mse is loss for gen (default True)",
    )


    parser.add_argument(
        "--mae",
        type=bool,
        default=False,
        help="mae is loss for gen (default False)",
    )

    parser.add_argument(
        "--cpgen",
        type=str,
        default="/home/tejas/experimentations/image_enhancement/checkpoints/gen.pth.tar",
        help="path for generator checkpoint",
    )

    parser.add_argument(
        "--cpdisc",
        type=str,
        default="/home/tejas/experimentations/image_enhancement/checkpoints/disc.pth.tar",
        help="path for discriminator checkpoint",
    )






    args = parser.parse_args()

    return args
