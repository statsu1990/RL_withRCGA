import RealCodedGeneticAlgorithm as rcga

import numpy as np
import matplotlib.pyplot as plt

import gym
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://github.com/openai/gym/blob/master/gym/core.py
import copy

from matplotlib import animation as anm
from datetime import datetime

class GymTest:
    def test():
        env = gym.make('CartPole-v0')
        env.reset()
        for _ in range(1000):
            env.render()
            env.step(env.action_space.sample())
            return

    def test2():
        env = gym.make('Pendulum-v0')
        for i_episode in range(20):
            observation = env.reset()
            for t in range(100):
                env.render()
                action = env.action_space.sample() # ここをカスタマイズするのが目的
                observation, reward, done, info = env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
        return

    def test3():
        env = gym.make('Pendulum-v0')
        for i_episode in range(20):
            observation = env.reset()
            #
            print('<episode, {0}>'.format(i_episode))
            for t in range(100):
                env.render()
                action = env.action_space.sample() # ここをカスタマイズするのが目的
                observation, reward, done, info = env.step(action)
                print(' {0}, {1}, {2}, {3}'.format(t, action, reward, observation))
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
        return

    def test4():
        env = gym.make('Pendulum-v0')
        observation = env.reset()
        print('最初')
        print(observation)
        print(env.env)
        
        #
        env_copy = copy.deepcopy(env)
        
        #
        print('start')
        for t in range(5):
            action = [1.0] # ここをカスタマイズするのが目的
            observation, reward, done, info = env.step(action)
            print(' {0}, {1}, {2}, {3}'.format(t, action, reward, observation))

        print('start copy')
        for t in range(5):
            action = [1.0] # ここをカスタマイズするのが目的
            observation, reward, done, info = env_copy.step(action)
            print(' {0}, {1}, {2}, {3}'.format(t, action, reward, observation))


        return

class Pendulum_withDNNandRCGA:
    def __init__(self, step_per_episode=100, discount_fact=0.9):
        # Pendulum-v0
        #
        # Observation
        # Type: Box(3)
        # Num	Observation	Min	Max
        # 0	cos(theta)	-1.0	1.0
        # 1	sin(theta)	-1.0	1.0
        # 2	theta dot	-8.0	8.0
        #
        # Actions
        # Type: Box(1)
        # Num	Action	Min	Max
        # 0	Joint effort	-2.0	2.0

        #
        self.action_num = 1
        self.observation_num = 3 # cos(theta), sin(theta), theta dot
        #
        self.observation_mean = np.array([0.0, 0.0, 0.0])
        self.observation_scale = np.array([1.0, 1.0, 8.0])
        self.action_mean = np.array([0.0])
        self.action_scale = np.array([2.0])


        # dnn model
        self.weights = None

        # genetic algorithm
        self.ga = None

        # gym env
        self.env = gym.make('Pendulum-v0')
        self.step_per_episode = step_per_episode
        self.discount_fact = discount_fact

        return

    # dnn
    def make_dnn_model(self, hidden_node_list):
        self.weights = []
        #
        for i_layer in range(len(hidden_node_list)+1):
            in_num = self.observation_num if i_layer==0 else hidden_node_list[i_layer-1]
            out_num = self.action_num if i_layer==len(hidden_node_list) else hidden_node_list[i_layer]
            # weight shape is (in, out)
            W = np.zeros((in_num, out_num))
            # bias shape is (out_num)
            b = np.zeros(out_num)
            #
            self.weights.append(W)
            self.weights.append(b)
        return

    def predict(self, x, weights=None):
        '''
        return dnn(x)
        '''
        w = self.weights.copy() if weights is None else weights

        # input scaling
        h = (x - self.observation_mean) / self.observation_scale
        
        # hidden layer
        for i_layer in range(int(len(w)/2)):
            # h_out = h_in * W + b
            h = np.dot(h, w[2*i_layer]) + w[2*i_layer+1]
            # activation func
            if i_layer != int(len(w)/2) - 1:
                h = self.__activation_func(h)

        # output
        h = self.__output_func(h)

        # output scaling
        h = h * self.action_scale + self.action_mean

        return h

    @staticmethod
    def __activation_func(x):
        # RELU
        a = np.maximum(0, x)
        # Leaky relu
        #a = np.maximum(0, x) + 0.1 * np.minimum(0, x)
        # SeLU
        #alpha = 1.6732632423543772848170429916717
        #scale = 1.0507009873554804934193349852946
        #a = scale * np.where(x >= 0.0, x, alpha * np.exp(x) - alpha)

        return a

    @staticmethod
    def __output_func(x):
        # tanh [-1, 1]
        y = np.tanh(x)
        # 
        #y = np.minimum(np.maximum(x, -1), 1)
        return y

    # ga
    def make_ga(self, initial_min=-1, initial_max=1, 
                population=20, 
                crossover_num=None, child_num=None, 
                initial_expantion_rate=None, learning_rate=None, 
                seed=None):
        # gene_num
        gene_num = 0
        for w in self.weights:
            gene_num += w.size
        #
        self.env.reset()
        #
        self.ga = rcga.RealCodecGA_JGG_AREX(gene_num=gene_num, 
                                            evaluation_func=self.evaluation_func_forGA, 
                                            initial_min=initial_min, initial_max=initial_max, 
                                            population=population, 
                                            crossover_num=crossover_num, child_num=child_num, 
                                            initial_expantion_rate=initial_expantion_rate, learning_rate=learning_rate, 
                                            seed=seed)
        return

    def __gene_to_weights(self, gene):
        '''
        gene = [flatten(w_layer1), b_layer1, flatten(w_layer2), b_layer2, ...]
        '''
        w = []
        #
        start_idx = 0
        end_idx = 0
        for iw in range(len(self.weights)):
            # weight, bias
            end_idx = start_idx + self.weights[iw].size
            w.append(np.reshape(gene[start_idx:end_idx], self.weights[iw].shape))
            start_idx = end_idx

        return w

    def evaluation_func_forGA(self, genes):
        '''
        rmse of y
        coef = bias_coefs[1:]
        bias = bias_coefs[0]
        '''
        #
        vec_f1 = np.vectorize(self.__gene_to_weights, signature='(m)->()')
        weightss = vec_f1(genes)
        #
        evals = []
        for w in weightss:
            evals.append(self.__eval_func(w))

        return np.array(evals)

    def __eval_func(self, weights=None, render=False, step_per_episode=None):
        '''
        run one episode
        return reward
        '''
        w = self.weights.copy() if weights is None else weights
        step_num = step_per_episode if step_per_episode is not None else self.step_per_episode

        #
        env_copy = copy.deepcopy(self.env)
        #
        frames = []
        #
        obsv = np.array(env_copy.reset())
        sum_reward = 0.0
        discounted_reward = 0.0
        for t in range(step_num):
            # 画面に出力するか
            if render:
                #env_copy.render()
                frames.append(env_copy.render(mode='rgb_array'))
            # get action
            action = self.predict(x=obsv, weights=w)
            #
            obsv, reward, done, info = env_copy.step(action)
            obsv = np.array(obsv)
            sum_reward += reward
            discounted_reward = reward + discounted_reward * self.discount_fact
        
        # save gif
        if render:
            self.__display_frames_as_gif(frames)

        return - discounted_reward

    def run(self, step_num, print_evaluation=True, print_fig=True):
        if print_evaluation:
            print('generation, best_evaluation, best_survive_evals, diversity')
        #
        best_evals = []
        for i in range(step_num):
            self.env.reset()
            best_survive_evals = self.ga.generation_step()
            best_evals.append(self.ga.best_evaluation)
            diversity = self.ga.calc_diversity()

            if print_evaluation:
                print('{0}, {1}, {2}, {3}'.format(i+1, self.ga.best_evaluation, best_survive_evals, diversity))

        # summary
        print('<best gene>')
        self.__summary(best_evals[-1], self.ga.best_gene)
        self.__render(self.ga.best_gene)
        self.__render(self.ga.best_gene)
        self.__render(self.ga.best_gene)
        self.__render(self.ga.best_gene)
        self.__render(self.ga.best_gene)
        self.__render(self.ga.best_gene)
        self.__render(self.ga.best_gene)
        self.__render(self.ga.best_gene)
        self.__render(self.ga.best_gene)
        print('<last gene>')
        self.__summary(best_survive_evals, self.ga.last_gene)
        self.__render(self.ga.last_gene)
        self.__render(self.ga.last_gene)
        self.__render(self.ga.last_gene)
        self.__render(self.ga.last_gene)
        self.__render(self.ga.last_gene)
        self.__render(self.ga.last_gene)
        self.__render(self.ga.last_gene)
        self.__render(self.ga.last_gene)
        self.__render(self.ga.last_gene)
        self.__render(self.ga.last_gene)

        # figure
        if print_fig:
            self.__conv_fig(best_evals, skip_display=2)

        return best_evals, self.ga.best_gene, self.ga.best_evaluation

    def __summary(self, eval, gene):
        # best
        print('<Genetic Algorithm>')
        print(' eval, {0}'.format(eval))
        print(' gene, {0}'.format(gene))
        return
    
    def __render(self, gene, step_per_episode=400):
        self.env.reset()
        weights = self.__gene_to_weights(gene)
        self.__eval_func(weights, True, step_per_episode=step_per_episode)
        return

    @staticmethod
    def __conv_fig(best_evals, skip_display=2):
        # fig
        display_start_idx = skip_display if len(best_evals) > skip_display else 0
        #display_start_idx = int((len(best_evals)+1)*(1-display_rate))
        #
        x = np.arange(display_start_idx+1, len(best_evals)+1)
        plt.plot(x, np.array(best_evals)[display_start_idx:], marker="o", markersize=2)
        plt.xlabel('generation')
        plt.ylabel('evaluation')
        plt.show()
        return

    @staticmethod
    def __display_frames_as_gif(frames):
        """
        Displays a list of frames as a gif, with controls
        """
        plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0),
                   dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')
 
        def animate(i):
            patch.set_data(frames[i])
 
        anim = anm.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
        
        anim.save('Pendulum-v0_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.gif', writer="pillow")

def test_Pendulum_withDNNandRCGA():
    '''
    step_per_episode = 200
    #
    hidden_node_list = [10, 5, 5, 5]
    #
    n = (np.sum(np.array([3]+hidden_node_list)*np.array(hidden_node_list+[1])) + np.sum(hidden_node_list+[1]))
    initial_min = -4
    initial_max = 4
    population = int(np.sqrt(n)*2)
    crossover_num = population
    child_num = 10*crossover_num
    initial_expantion_rate = 1.0
    learning_rate = 0.0
    ga_seed = 11
    #
    step_num=100
    '''
    
    step_per_episode = 200
    #
    #hidden_node_list = [10, 5]
    hidden_node_list = [10, 5, 5]
    #
    n = (np.sum(np.array([3]+hidden_node_list)*np.array(hidden_node_list+[1])) + np.sum(hidden_node_list+[1]))
    initial_min = -4
    initial_max = 4
    population = 30
    crossover_num = population
    child_num = 10*crossover_num
    initial_expantion_rate = 1.0
    learning_rate = 0.0
    ga_seed = 11
    #
    step_num=100
    
    #
    vr = Pendulum_withDNNandRCGA(step_per_episode=step_per_episode)
    #
    vr.make_dnn_model(hidden_node_list=hidden_node_list)
    vr.make_ga(initial_min=initial_min, initial_max=initial_max, 
                population=population, 
                crossover_num=crossover_num, child_num=child_num, 
                initial_expantion_rate=initial_expantion_rate, learning_rate=learning_rate, 
                seed=ga_seed)

    _, _, best_eval = vr.run(step_num=step_num, print_evaluation=True, print_fig=True)

    return


if __name__ == '__main__':
    test_Pendulum_withDNNandRCGA()
