from GDWCT import GDWCT
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GDWCT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--dataset', type=str, default='male2female', help='dataset_name')
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')

    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations')
    # The total number of iterations is [epoch * iteration]

    # See here for accurate hyper-parameters
    # https://github.com/WonwoongCho/GDWCT/blob/master/configs/config.yaml

    parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=50000, help='The number of ckpt_save_freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_start_epoch', type=int, default=10, help='decay start epoch')
    parser.add_argument('--decat_step_epoch', type=int, default=5, help='decay step epoch')


    parser.add_argument('--num_style', type=int, default=3, help='number of styles to sample')
    parser.add_argument('--direction', type=str, default='a2b', help='direction of style guided image translation')
    parser.add_argument('--guide_img', type=str, default='guide.png', help='Style guided image translation')

    parser.add_argument('--gan_type', type=str, default='lsgan', help='GAN loss type [gan / lsgan / hinge]')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--gan_w', type=int, default=1, help='weight of adversarial loss')
    parser.add_argument('--recon_x_w', type=int, default=10, help='weight of image reconstruction loss')
    parser.add_argument('--recon_s_w', type=int, default=1, help='weight of style reconstruction loss')
    parser.add_argument('--recon_c_w', type=int, default=1, help='weight of content reconstruction loss')
    parser.add_argument('--recon_x_cyc_w', type=int, default=10, help='weight of explicit style augmented cycle consistency loss')
    parser.add_argument('--lambda_w', type=float, default=0.001, help='weight of whitening')
    parser.add_argument('--lambda_c', type=int, default=10, help='weight of coloring')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--style_dim', type=int, default=256, help='length of style code')
    parser.add_argument('--n_res', type=int, default=4, help='number of residual blocks in content encoder/decoder')
    parser.add_argument('--group_num', type=int, default=8, help='group num')

    parser.add_argument('--n_dis', type=int, default=4, help='number of discriminator layer')
    parser.add_argument('--n_scale', type=int, default=3, help='number of scales')
    parser.add_argument('--sn', type=str2bool, default=False, help='using spectral norm')

    parser.add_argument('--img_h', type=int, default=256, help='The size of image hegiht')
    parser.add_argument('--img_w', type=int, default=256, help='The size of image width')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = GDWCT(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train':
            # launch the graph in a session
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test':
            gan.style_guide_test()
            print(" [*] Guide test finished!")


if __name__ == '__main__':
    main()