#!/usr/bin/env python
import tensorflow as tf
import gym
import roboschool

envName = 'Humanoid-v2'

env = gym.make(envName)
sDim = env.observation_space.shape[0]
#env.render(mode='human')
state = env.reset()

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./models/'+envName+'Agent.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./models'))
    graph = tf.get_default_graph()
    actor = graph.get_tensor_by_name('actor/actor/Tanh:0')
    s = graph.get_tensor_by_name('s:0')
    cummulativeReward = 0.
    done = False
    for i in range(10):
        env.reset()
        env.render()
        while not done:
            action = sess.run(actor, feed_dict={s:state.reshape(1,sDim)})[0]
            state, reward, doneReport, ifo = env.step(action)
            cummulativeReward += reward
            env.render()
            if doneReport:
                done = True

