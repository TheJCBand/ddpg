#!/usr/bin/env python

# Import packages
import numpy as np
import tensorflow as tf
import gym
import roboschool
from collections import deque
import random
import matplotlib.pyplot as plt

# User parameters
envName = 'RoboschoolInvertedPendulumSwingup-v1' # Envirornment
nCriticUnits = [400,300] # Number of nodes in each hidden layer of actor network
nActorUnits = [400,300] # Number of nodes in each hidden layer of critic network
nEpisodes = 25000 # Number of episodes to train for
sigma = 0.2 # Orenstein-Uhlenbeck parameters
theta = 0.15
minibatchSize = 64 
gamma = 0.99 # Discount factor
tau = 0.001 # Parameter for updating target networks
replayBufferSize = 10**6
criticLearningRate = 10**-3 
actorLearningRate = 10**-4
decay = 0.018 # L2 weight decay for critic
renderDuringTraining = False
plotLearningCurve = False # Plot cummulative reward per episode during training

# Make environment
env = gym.make(envName)
aDim = env.action_space.shape[0]
sDim = env.observation_space.shape[0]

# Define placeholders for Tensorflow graph inputs
s = tf.placeholder(tf.float32, shape=(None,sDim),name='s')
a = tf.placeholder(tf.float32, shape=(None,aDim))
sTarget = tf.placeholder(tf.float32, shape=(None,sDim))
aTarget = tf.placeholder(tf.float32, shape=(None,aDim))
r = tf.placeholder(tf.float32, shape=(None,1))

# Define L2 regularizer
reg = tf.contrib.layers.l2_regularizer(decay)

# Critic network
with tf.variable_scope('critic'):
    critic = tf.layers.batch_normalization(s)
    critic = tf.layers.dense(critic,nCriticUnits[0],activation=tf.nn.relu,kernel_regularizer=reg)
    critic = tf.layers.batch_normalization(critic)
    critic = tf.keras.layers.concatenate([critic,a])
    critic = tf.layers.dense(critic,nCriticUnits[1],activation=tf.nn.relu,kernel_regularizer=reg)
    critic = tf.layers.dense(critic,1,kernel_regularizer=reg)

# Critic target network
with tf.variable_scope('criticTarget'):
    criticTarget = tf.layers.batch_normalization(sTarget,trainable=False)
    criticTarget = tf.layers.dense(criticTarget,nCriticUnits[0],activation=tf.nn.relu,kernel_regularizer=reg,trainable=False)
    criticTarget = tf.layers.batch_normalization(criticTarget,trainable=False)
    criticTarget = tf.keras.layers.concatenate([criticTarget,aTarget],trainable=False)
    criticTarget = tf.layers.dense(criticTarget,nCriticUnits[1],activation=tf.nn.relu,kernel_regularizer=reg,trainable=False)
    criticTarget = tf.layers.dense(criticTarget,1,kernel_regularizer=reg,trainable=False)

# Actor network
with tf.variable_scope('actor'):
    actor = tf.layers.batch_normalization(s)
    actor = tf.layers.dense(actor,nActorUnits[0],activation=tf.nn.relu)
    actor = tf.layers.batch_normalization(actor)
    actor = tf.layers.dense(actor,nActorUnits[1],activation=tf.nn.relu)
    actor = tf.layers.batch_normalization(actor)
    actor = tf.layers.dense(actor,aDim,activation=tf.nn.tanh,name='actor')

# Actor target network
with tf.variable_scope('actorTarget'):
    actorTarget = tf.layers.batch_normalization(sTarget)
    actorTarget = tf.layers.dense(actorTarget,nActorUnits[0],activation=tf.nn.relu,trainable=False) 
    actorTarget = tf.layers.batch_normalization(actorTarget)
    actorTarget = tf.layers.dense(actorTarget,nActorUnits[1],activation=tf.nn.relu,trainable=False)
    actorTarget = tf.layers.batch_normalization(actorTarget)
    actorTarget = tf.layers.dense(actorTarget,aDim,activation=tf.nn.tanh,trainable=False)

# Get weight variables for each network
criticWeights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='critic')
criticTargetWeights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='criticTarget')
actorWeights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='actor')
actorTargetWeights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='actorTarget')

# Define the training op for each network
# This is needed for batch normalization
updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(updateOps):
    
    # Train Critic
    L = tf.losses.mean_squared_error(r+gamma*criticTarget, critic)
    trainCritic = tf.train.AdamOptimizer(criticLearningRate).minimize(L)
    
    # Train Actor
    # Gradient of critic network w.r.t. action
    criticActionGrad = tf.gradients(critic,a)[0]
    # Gradient of policy performance w.r.t. actor network parameters.  
    # tf.gradients multiplies gradient by grad_ys, so this line applies the chain rule
    policyGrad = tf.gradients(actor,actorWeights,grad_ys=criticActionGrad)
    # Negative learning rate is used for gradient ascent.  apply_gradients takes one step of gradient descent.
    trainActor = tf.train.GradientDescentOptimizer(-actorLearningRate).apply_gradients(zip(policyGrad,actorWeights))
    
# Train Targets
trainCriticTarget = [tf.assign(criticTargetWeight, tau*criticWeight+(1-tau)*criticTargetWeight) for criticTargetWeight, criticWeight in zip(criticTargetWeights, criticWeights)]
trainActorTarget = [tf.assign(actorTargetWeight, tau*actorWeight+(1-tau)*actorTargetWeight) for actorTargetWeight, actorWeight in zip(actorTargetWeights, actorWeights)]

# Function for plotting performance curve
def performanceCurve(cummulativeRewardList, realTime=False):
    plt.cla()
    plt.plot(cummulativeRewardList)
    plt.xlabel('episode')
    plt.ylabel('cummulative reward')
    plt.title(envName+' Performance Curve')
    if realTime:
        plt.pause(.0001)

# Initialize Tensorflow session
sess = tf.Session()
# Initialize all variables
sess.run(tf.initializers.global_variables())

# Initialize target weights to same as non-targets
sess.run([tf.assign(criticTargetWeight, criticWeight) for criticTargetWeight, criticWeight in zip(criticTargetWeights, criticWeights)])
sess.run([tf.assign(actorTargetWeight, actorWeight) for actorTargetWeight, actorWeight in zip(actorTargetWeights, actorWeights)])

# Initialize replay buffer
R = deque()
# Initialize a list to store cummulative rewards from each episode.  This is used for plotting the learning curve.
cummulativeRewardList = []
# Training loop
for episode in range(nEpisodes):
    # Initialize Ornstein-Uhlenbeck noise
    noise = 0.
    # Initialize environment
    state = env.reset()
    # Render
    if renderDuringTraining:
        env.render()
    done = False
    # Initialize cummulative reward for this episode
    cummulativeReward = 0.
    while not done:
        # Choose action
        action = sess.run(actor, feed_dict={s:state.reshape((1,sDim))})[0]
        # Update Ornstein-Uhlenbeck noise
        noise += -theta*noise + sigma*np.array([random.random() for i in range(aDim)])
        # Apply action with noise to environment
        stateNext, reward, doneReport, info = env.step(action+noise)
        # Render
        if renderDuringTraining:
            env.render()
        # Update cummulative reward (used for plotting performance curve)
        cummulativeReward += reward
        # Add this transition to the replay buffer
        R.append({'s':state, 'a':action+noise, 'r':[reward], 'sNext':stateNext})
        if len(R) > replayBufferSize:
            R.popleft()
        # Sample a minibatch from the replay buffer
        if len(R) < minibatchSize:
            minibatch = random.sample(R, len(R))
        else:
            minibatch = random.sample(R, minibatchSize)
        sBatch = [sample['s'] for sample in minibatch]
        aBatch = [sample['a'] for sample in minibatch]
        rBatch = [sample['r'] for sample in minibatch]
        sNextBatch = [sample['sNext'] for sample in minibatch]
        # Calculate actor target output for critic loss function
        aTargetBatch = sess.run(actorTarget, feed_dict={sTarget:sNextBatch})
        # Update each network using their respective training ops
        sess.run(trainCritic,feed_dict={r:rBatch,sTarget:sNextBatch,aTarget:aTargetBatch,s:sBatch,a:aBatch})
        sess.run(trainActor,feed_dict={s:sBatch,a:aBatch})
        sess.run(trainCriticTarget)
        sess.run(trainActorTarget)
        # Advance state variable
        state = stateNext
        # Do these things at the end of the episode
        if doneReport:
            cummulativeRewardList.append(cummulativeReward)
            print('Episode {} cummulative reward: {}'.format(episode, cummulativeReward))
            # Update the performance plot
            if plotLearningCurve:
                performanceCurve(cummulativeRewardList, realTime=True)
            # Save network weights every 200 episodes
            if episode % 200 == 0:
                print('Saving models')
                tf.train.Saver().save(sess, './models/'+envName+'Agent')
                performanceCurve(cummulativeRewardList)
                plt.savefig('./models/'+envName+'PerformanceCurve.pdf')
            done = True

