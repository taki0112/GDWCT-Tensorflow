from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch

class GDWCT(object) :
    def __init__(self, sess, args):
        self.model_name = 'GDWCT'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.iteration = args.iteration

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.num_style = args.num_style # for test
        self.guide_img = args.guide_img
        self.direction = args.direction

        self.img_h = args.img_h
        self.img_w = args.img_w
        self.img_ch = args.img_ch

        self.init_lr = args.lr
        self.decay_flag = args.decay_flag
        self.decay_start_epoch = args.decay_start_epoch
        self.decay_step_epoch = args.decay_step_epoch

        self.ch = args.ch

        self.phase = args.phase

        """ Weight """
        self.gan_w = args.gan_w
        self.recon_x_w = args.recon_x_w
        self.recon_s_w = args.recon_s_w
        self.recon_c_w = args.recon_c_w
        self.recon_x_cyc_w = args.recon_x_cyc_w
        self.lambda_w = args.lambda_w
        self.lambda_c = args.lambda_c

        """ Generator """
        self.n_res = args.n_res
        self.group_num = args.group_num

        self.style_dim = args.style_dim

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_scale = args.n_scale
        self.sn = args.sn

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)
        print("# style in test phase : ", self.num_style)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)
        print("# Style dimension : ", self.style_dim)
        print("# group number : ", self.group_num)


        print()

        print("##### Discriminator #####")
        print("# Discriminator layer : ", self.n_dis)
        print("# Multi-scale Dis : ", self.n_scale)
        print("# spectral norm : ", self.sn)

        print()

    ##################################################################################
    # Encoder and Decoders
    ##################################################################################

    def content_encoder(self, x, reuse=False, scope='content_encoder'):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', sn=self.sn, scope='conv_0')
            x = instance_norm(x, scope='ins_0')
            x = relu(x)

            for i in range(2) :
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_'+str(i+1))
                x = instance_norm(x, scope='ins_'+str(i+1))
                x = relu(x)

                channel = channel * 2

            for i in range(self.n_res) :
                x = resblock(x, channel, sn=self.sn, scope='resblock_'+str(i))

            return x

    def style_encoder(self, x, reuse=False, scope='style_encoder'):
        # use group_norm
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', sn=self.sn, scope='conv_0')
            x = group_norm(x, groups=self.group_num, scope='group_norm_0')
            x = relu(x)

            for i in range(2) :
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_'+str(i+1))
                x = group_norm(x, groups=self.group_num, scope='group_norm_' + str(i+1))
                x = relu(x)

                channel = channel * 2

            for i in range(2) :
                x = conv(x, channel, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_'+str(i+1+2))
                x = group_norm(x, groups=self.group_num, scope='group_norm_' + str(i+1+2))
                x = relu(x)

            x = global_avg_pooling(x) # global average pooling

            return x

    def generator(self, content, style, reuse=False, scope="decoder"):
        channel = self.style_dim
        with tf.variable_scope(scope, reuse=reuse) :
            x = content
            U_list = [] # for regularization

            for i in range(self.n_res) :
                if i == 0:
                    x, U = self.WCT(content, style, sn=self.sn, scope='front_WCT_' + str(i))
                    U_list.append(U)

                x = no_norm_resblock(x, channel, sn=self.sn, scope='no_norm_resblock_' + str(i))

                x, U = self.WCT(x, style, sn=self.sn, scope='back_WCT_' + str(i))
                U_list.append(U)

            for i in range(2) :
                x = up_sample_nearest(x, scale_factor=2)
                x = conv(x, channel//2, kernel=5, stride=1, pad=2, pad_type='reflect', sn=self.sn, scope='conv_'+str(i))
                x = layer_norm(x, scope='layer_norm_' + str(i))
                x = relu(x)

                channel = channel // 2

            x = conv(x, channels=self.img_ch, kernel=7, stride=1, pad=3, pad_type='reflect', sn=self.sn, scope='G_logit')
            x = tanh(x)

            return x, U_list

    def WCT(self, content, style, sn=False, scope='wct'):
        with tf.variable_scope(scope) :
            mu = self.MLP(style, sn=sn, scope='MLP_mu')
            ct = self.MLP(style, sn=sn, scope='MLP_CT')

            alpha = tf.get_variable('alpha', shape=[1], initializer=tf.constant_initializer(0.6), constraint=lambda v: tf.clip_by_value(v, 0.0, 1.0))

            x, U = GDWCT_block(content, ct, style_mu=mu, group_num=self.group_num)
            x = alpha * x + (1 - alpha) * content

            return x, U

    def MLP(self, x, sn=False, scope='MLP'):
        channel = self.style_dim
        with tf.variable_scope(scope) :

            for i in range(2) :
                x = fully_connected(x, channel, sn=sn, scope='linear_' + str(i))
                x = lrelu(x, 0.01)


            x = fully_connected(x, channel, sn=sn, scope='logit')

            x = tf.reshape(x, shape=[-1, 1, channel])


            return x


    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        D_logit = []
        with tf.variable_scope(scope, reuse=reuse) :
            for scale in range(self.n_scale) :
                channel = self.ch
                x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='ms_' + str(scale) + '_conv_0')
                x = lrelu(x, 0.01)

                for i in range(1, self.n_dis):
                    x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='ms_' + str(scale) +'_conv_' + str(i))
                    x = lrelu(x, 0.01)

                    channel = channel * 2

                x = conv(x, channels=1, kernel=1, stride=1, sn=self.sn, scope='ms_' + str(scale) + '_D_logit')
                D_logit.append(x)

                x_init = down_sample(x_init)

            return D_logit

    ##################################################################################
    # Model
    ##################################################################################

    def encoder_A(self, x_A, reuse=False):
        style_A = self.style_encoder(x_A, reuse=reuse, scope='style_encoder_A')
        content_A = self.content_encoder(x_A, reuse=reuse, scope='content_encoder_A')

        return content_A, style_A

    def encoder_B(self, x_B, reuse=False):
        style_B = self.style_encoder(x_B, reuse=reuse, scope='style_encoder_B')
        content_B = self.content_encoder(x_B, reuse=reuse, scope='content_encoder_B')

        return content_B, style_B

    def decoder_A(self, content_B, style_A, reuse=False):
        x_ba, U_style_A = self.generator(content=content_B, style=style_A, reuse=reuse, scope='decoder_A')

        return x_ba, U_style_A

    def decoder_B(self, content_A, style_B, reuse=False):
        x_ab, U_style_B = self.generator(content=content_A, style=style_B, reuse=reuse, scope='decoder_B')

        return x_ab, U_style_B

    def discriminate_real(self, x_A, x_B):
        real_A_logit = self.discriminator(x_A, scope="discriminator_A")
        real_B_logit = self.discriminator(x_B, scope="discriminator_B")

        return real_A_logit, real_B_logit

    def discriminate_fake(self, x_ba, x_ab):
        fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
        fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

        return fake_A_logit, fake_B_logit

    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Input Image"""
        Image_Data_Class = ImageData(self.img_h, self.img_w, self.img_ch, self.augment_flag)

        trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
        trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)

        gpu_device = '/gpu:0'

        trainA = trainA.\
            apply(shuffle_and_repeat(self.dataset_num)). \
            apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)). \
            apply(prefetch_to_device(gpu_device, None))

        trainB = trainB. \
            apply(shuffle_and_repeat(self.dataset_num)). \
            apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)). \
            apply(prefetch_to_device(gpu_device, None))
        # When using dataset.prefetch, use buffer_size=None to let it detect optimal buffer size

        trainA_iterator = trainA.make_one_shot_iterator()
        trainB_iterator = trainB.make_one_shot_iterator()


        self.domain_A = trainA_iterator.get_next()
        self.domain_B = trainB_iterator.get_next()


        """ Define Encoder, Generator, Discriminator """

        # encode
        content_a, style_a = self.encoder_A(self.domain_A)
        content_b, style_b = self.encoder_B(self.domain_B)

        # decode (cross domain)
        x_ba, U_A = self.decoder_A(content_B=content_b, style_A=style_a)
        x_ab, U_B = self.decoder_B(content_A=content_a, style_B=style_b)

        # decode (within domain)
        x_aa, _ = self.decoder_A(content_B=content_a, style_A=style_a, reuse=True)
        x_bb, _ = self.decoder_B(content_A=content_b, style_B=style_b, reuse=True)


        # encode again
        content_ba, style_ba = self.encoder_A(x_ba, reuse=True)
        content_ab, style_ab = self.encoder_B(x_ab, reuse=True)

        # decode again (if needed)
        x_aba, _ = self.decoder_A(content_B=content_ab, style_A=style_ba, reuse=True)
        x_bab, _ = self.decoder_B(content_A=content_ba, style_B=style_ab, reuse=True)

        real_A_logit, real_B_logit = self.discriminate_real(self.domain_A, self.domain_B)
        fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)

        """ Define Loss """
        G_adv_A = self.gan_w * generator_loss(self.gan_type, fake_A_logit)
        G_adv_B = self.gan_w * generator_loss(self.gan_type, fake_B_logit)

        D_adv_A = self.gan_w * discriminator_loss(self.gan_type, real_A_logit, fake_A_logit)
        D_adv_B = self.gan_w * discriminator_loss(self.gan_type, real_B_logit, fake_B_logit)


        recon_style_A = self.recon_s_w * L1_loss(style_ba, style_a)
        recon_style_B = self.recon_s_w * L1_loss(style_ab, style_b)

        recon_content_A = self.recon_c_w * L1_loss(content_ab, content_a)
        recon_content_B = self.recon_c_w * L1_loss(content_ba, content_b)

        cyc_recon_A = self.recon_x_cyc_w * L1_loss(x_aba, self.domain_A)
        cyc_recon_B = self.recon_x_cyc_w * L1_loss(x_bab, self.domain_B)

        recon_A = self.recon_x_w * L1_loss(x_aa, self.domain_A) # reconstruction
        recon_B = self.recon_x_w * L1_loss(x_bb, self.domain_B) # reconstruction

        whitening_A, coloring_A = group_wise_regularization(deep_whitening_transform(content_a), U_A, self.group_num)
        whitening_B, coloring_B = group_wise_regularization(deep_whitening_transform(content_b), U_B, self.group_num)

        whitening_A = self.lambda_w * whitening_A
        whitening_B = self.lambda_w * whitening_B

        coloring_A = self.lambda_c * coloring_A
        coloring_B = self.lambda_c * coloring_B

        G_reg_A = regularization_loss('decoder_A') + regularization_loss('encoder_A')
        G_reg_B = regularization_loss('decoder_B') + regularization_loss('encoder_B')

        D_reg_A = regularization_loss('discriminator_A')
        D_reg_B = regularization_loss('discriminator_B')


        Generator_A_loss = G_adv_A + \
                           recon_A + \
                           recon_style_A + \
                           recon_content_A + \
                           cyc_recon_B + \
                           whitening_A + \
                           coloring_A + \
                           G_reg_A

        Generator_B_loss = G_adv_B + \
                           recon_B + \
                           recon_style_B + \
                           recon_content_B + \
                           cyc_recon_A + \
                           whitening_B + \
                           coloring_B + \
                           G_reg_B

        Discriminator_A_loss = D_adv_A + D_reg_A
        Discriminator_B_loss = D_adv_B + D_reg_B

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'decoder' in var.name or 'encoder' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)

        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_A_adv_loss = tf.summary.scalar("G_A_adv_loss", G_adv_A)
        self.G_A_style_loss = tf.summary.scalar("G_A_style_loss", recon_style_A)
        self.G_A_content_loss = tf.summary.scalar("G_A_content_loss", recon_content_A)
        self.G_A_cyc_loss = tf.summary.scalar("G_A_cyc_loss", cyc_recon_A)
        self.G_A_identity_loss = tf.summary.scalar("G_A_identity_loss", recon_A)
        self.G_A_whitening_loss = tf.summary.scalar("G_A_whitening_loss", whitening_A)
        self.G_A_coloring_loss = tf.summary.scalar("G_A_coloring_loss", coloring_A)

        self.G_B_adv_loss = tf.summary.scalar("G_B_adv_loss", G_adv_B)
        self.G_B_style_loss = tf.summary.scalar("G_B_style_loss", recon_style_B)
        self.G_B_content_loss = tf.summary.scalar("G_B_content_loss", recon_content_B)
        self.G_B_cyc_loss = tf.summary.scalar("G_B_cyc_loss", cyc_recon_B)
        self.G_B_identity_loss = tf.summary.scalar("G_B_identity_loss", recon_B)
        self.G_B_whitening_loss = tf.summary.scalar("G_B_whitening_loss", whitening_B)
        self.G_B_coloring_loss = tf.summary.scalar("G_B_coloring_loss", coloring_B)

        self.alpha_var = []
        for var in tf.trainable_variables():
            if 'alpha' in var.name:
                self.alpha_var.append(tf.summary.histogram(var.name, var))
                self.alpha_var.append(tf.summary.scalar(var.name, tf.reduce_max(var)))

        G_summary_list = [self.G_A_adv_loss,
                                        self.G_A_style_loss, self.G_A_content_loss,
                                        self.G_A_cyc_loss, self.G_A_identity_loss,
                                        self.G_A_whitening_loss, self.G_A_coloring_loss,
                                        self.G_A_loss,

                                        self.G_B_adv_loss,
                                        self.G_B_style_loss, self.G_B_content_loss,
                                        self.G_B_cyc_loss, self.G_B_identity_loss,
                                        self.G_B_whitening_loss, self.G_B_coloring_loss,
                                        self.G_B_loss,

                                        self.all_G_loss]

        G_summary_list.extend(self.alpha_var)

        self.G_loss = tf.summary.merge(G_summary_list)
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        """ Image """
        self.fake_A = x_ba
        self.fake_B = x_ab

        self.real_A = self.domain_A
        self.real_B = self.domain_B

        """ Test """

        """ Guided Image Translation """
        self.content_image = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, self.img_ch], name='content_image')
        self.style_image = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, self.img_ch], name='guide_style_image')

        if self.direction == 'a2b' :
            guide_content_A, _ = self.encoder_A(self.content_image, reuse=True)
            _, guide_style_B = self.encoder_B(self.style_image, reuse=True)

            self.guide_fake_B, _ = self.decoder_B(content_A=guide_content_A, style_B=guide_style_B, reuse=True)

        else :
            guide_content_B, _ = self.encoder_B(self.content_image, reuse=True)
            _, guide_style_A = self.encoder_A(self.style_image, reuse=True)

            self.guide_fake_A, _ = self.decoder_A(content_B=guide_content_B, style_A=guide_style_A, reuse=True)



    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=20)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        lr = self.init_lr

        for epoch in range(start_epoch, self.epoch):

            if self.decay_flag and self.decay_start_epoch > epoch :
                lr = self.init_lr * pow(0.5, (epoch - self.decay_start_epoch) // self.decay_step_epoch)

            for idx in range(start_batch_id, self.iteration):

                train_feed_dict = {
                    self.lr : lr
                }

                # Update D
                _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G
                batch_A_images, batch_B_images, fake_A, fake_B, _, g_loss, summary_str = self.sess.run([self.real_A, self.real_B, self.fake_A, self.fake_B, self.G_optim, self.Generator_loss, self.G_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                if np.mod(idx+1, self.print_freq) == 0 :
                    save_images(batch_A_images, [self.batch_size, 1],
                                './{}/real_A_{:03d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx+1))
                    # save_images(batch_B_images, [self.batch_size, 1],
                    #             './{}/real_B_{:03d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx+1))

                    # save_images(fake_A, [self.batch_size, 1],
                    #             './{}/fake_A_{:03d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx+1))
                    save_images(fake_B, [self.batch_size, 1],
                                './{}/fake_B_{:03d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx+1))

                if np.mod(counter - 1, self.save_freq) == 0 :
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):

        n_dis = str(self.n_scale) + 'multi_' + str(self.n_dis) + 'dis'

        sn = ''

        if self.sn :
            sn = '_sn'



        return "{}_{}_{}_{}_{}adv_{}style_{}content_{}identity_{}cyc_{}color_{}white{}".format(self.model_name, self.dataset_name, self.gan_type,
                                                           n_dis,
                                                           self.gan_w, self.recon_s_w, self.recon_c_w,
                                                           self.recon_x_w, self.recon_x_cyc_w,
                                                           self.lambda_c, self.lambda_w, sn)
    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.\
                model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def style_guide_test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        style_file = load_test_data(self.guide_img, size_h=self.img_h, size_w=self.img_w)

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir, 'guide')
        check_folder(self.result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        if self.direction == 'a2b' :
            for sample_file in test_A_files:  # A -> B
                print('Processing A image: ' + sample_file)
                sample_image = load_test_data(sample_file, size_h=self.img_h, size_w=self.img_w)
                image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

                fake_img = self.sess.run(self.guide_fake_B, feed_dict={self.content_image: sample_image, self.style_image : style_file})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../../..' + os.path.sep + sample_file), self.img_w, self.img_h))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../../..' + os.path.sep + image_path), self.img_w, self.img_h))
                index.write("</tr>")

        else :
            for sample_file in test_B_files:  # B -> A
                print('Processing B image: ' + sample_file)
                sample_image = np.asarray(load_test_data(sample_file, size_h=self.img_h, size_w=self.img_w))
                image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

                fake_img = self.sess.run(self.guide_fake_A, feed_dict={self.content_image: sample_image, self.style_image : style_file})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../../..' + os.path.sep + sample_file), self.img_w, self.img_h))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../../..' + os.path.sep + image_path), self.img_w, self.img_h))
                index.write("</tr>")
        index.close()
