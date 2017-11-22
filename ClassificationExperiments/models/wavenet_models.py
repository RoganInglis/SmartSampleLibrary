from models import BaseModel
import tensorflow as tf
from models import utils


class WavenetModel(BaseModel):
    def set_model_props(self, config):
        self.wavenet_checkpoint_path = "./wavenet_checkpoints/wavenet-ckpt/model.ckpt-200000"
        self.sample_length = 64000
        self.num_classes = config['num_classes']

    def get_best_config(self):
        # This function is here to be overridden completely.
        # It returns a dictionary used to update the initial configuration (see __init__)
        return {}

    @staticmethod
    def get_random_config(fixed_params={}):
        # Why static? Because you want to be able to pass this function to other processes
        # so they can independently generate random configuration of the current model
        raise Exception('The get_random_config function must be overriden by the agent')

    def build_graph(self, graph):
        with graph.as_default():
            # Built wavenet graph
            wavenet = utils.build_wavenet(batch_size=self.batch_size, sample_length=self.sample_length)
            self.placeholders = {'inputs': wavenet['X'],
                                 'labels': tf.placeholder(tf.int32, shape=[self.batch_size, self.num_classes])}
            self.wavenet_saver = tf.train.Saver()  # Create saver here so we can restore only wavenet weights from .ckpt
            self.sample_embedding = wavenet['encoding']

            # Build net to classify embedding from wavenet
            self.logits, self.preds = utils.classification_net(self.sample_embedding, self.num_classes)

            self.prediction = tf.argmax(self.preds, axis=1)

            # Define softmax cross entropy loss
            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.placeholders['labels'], logits=self.logits)

            # Set up optimiser and train op
            self.optimiser = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimiser.minimize(self.loss)

            # Add Tensorboard ops
            tf.summary.histogram('sample_embedding', self.sample_embedding)
            tf.summary.scalar('loss', self.loss)
            self.train_summary = tf.summary.merge_all()

        return graph

    def infer(self, audio_input):
        raise Exception('The infer function must be overriden by the agent')

    def test(self):
        raise Exception('The test function must be overriden by the agent')

    def learn_from_epoch(self):
        # I like to separate the function to train per epoch and the function to train globally
        print('test')
        raise Exception('The learn_from_epoch function must be overriden by the agent')

    def init(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overidden cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            self.sess.run(self.init_op)
            self.wavenet_saver.restore(self.sess, self.wavenet_checkpoint_path)
        else:
            if self.config['debug']:
                print('Loading the model from folder: %s' % self.result_dir)
            self.sess.run(self.init_op)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)