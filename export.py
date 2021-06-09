import os
import tensorflow as tf
from custom_model import toy_model

# trained_checkpoint_prefix = '/linguistics/ethan/DL_Prototype/models/example_ckpt/ckpt-40'
# export_dir = os.path.join('/linguistics/ethan/DL_Prototype/models/example_ckpt/export/toy_model_ckpt2pb')

# graph = tf.Graph()
# with tf.compat.v1.Session(graph=graph) as sess:
#     # Restore from checkpoint
#     loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.index')
#     loader.restore(sess, trained_checkpoint_prefix)
#
#     # Export checkpoint to SavedModel
#     builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
#     builder.add_meta_graph_and_variables(sess,
#                                          [tf.saved_model.TRAINING, tf.saved_model.SERVING],
#                                          strip_default_attrs=True)
#     builder.save()
#

def export(model, ckpt_dir, SavedModel_dir, optimizer):

    print("Restoring checkpoint...")
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                               optimizer=optimizer,
                               net=model)

    ckpt.restore(ckpt_dir)
    print("Saving model as .pb")
    model.save(SavedModel_dir)


if __name__ == "__main__":

    my_model = toy_model()
    ckpt_dir = "/linguistics/ethan/DL_Prototype/models/example_ckpt/ckpt-60"
    SavedModel_dir = "/linguistics/ethan/DL_Prototype/models/example_ckpt/export/toy_model_ckpt2pb"
    optimizer = tf.keras.optimizers.Adam()
    export(my_model, ckpt_dir, SavedModel_dir, optimizer)
