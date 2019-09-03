#%%
import gym 
env=gym.make("CartPole-v0")
obs=env.reset()
obs

#%%
env.render()

#%%
img=env.render(mode='rgb_array')
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()


#%%
env.action_space

#%%
action=1
obs,reward,done,info=env.step(action)
obs

#%%
done

#%%
reward

#%%
def basic_policy(obs):
    angle=obs[2]
    if angle < 0:
        return 0
    else:
        return 1
total=[]
for episode in range(500):
    episide_reward=0
    obs=env.reset()
    for step in range(1000):
        action=basic_policy(obs)
        obs,reward,done,info=env.step(action)
        episide_reward+=reward
        if done:
            break
    total.append(episide_reward)
total
#%%
import tensorflow as tf
n_inputs=4
n_hidden=4
n_outputs=1
initializer = tf.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs)
outputs = tf.nn.sigmoid(logits)  # probability of action 0 (left)
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

y = 1. - tf.to_float(action)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

n_iterations = 250 # 训练迭代次数
n_max_steps = 1000 # 每一次的最大步长
n_games_per_update = 10 # 每迭代十次训练一次策略网络
save_iterations = 10 # 每十次迭代保存模型
discount_rate = 0.95
env = gym.make("CartPole-v0")

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        print("\rIteration: {}".format(iteration), end="")
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)
        feed_dict = {}
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            saver.save(sess, "./my_policy_net_pg.ckpt")

#%%
import numpy as np
def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    print('discount',discounted_rewards)
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        print('cumulative_rewards',cumulative_rewards)
        discounted_rewards[step] = cumulative_rewards
        print('discount',discounted_rewards)
    return discounted_rewards
discount_rewards([10, 0, -50], discount_rate=0.8)

#%%
np.empty(3)

#%%
