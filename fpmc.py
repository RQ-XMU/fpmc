# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:21:25 2019

@author: Administrator
"""
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import math
import random
import re
import os
import glob
import sys
from time import time
import evaluation_val

class FPMC(object):

    def __init__(self, reg = 0.0025, learning_rate = 0.05, annealing=1., init_sigma = 1, k_cf = 100, k_mc = 100, **kwargs):
        self.name = 'FPMC'
        self.reg = reg
        self.learning_rate = learning_rate  # self.learning_rate will change due to annealing.
        self.init_learning_rate = learning_rate  # self.init_learning_rate keeps the original value (for filename)
        self.annealing_rate = annealing
        self.init_sigma = init_sigma
        self.metrics = {'recall': {'direction': 1},
                        'sps': {'direction': 1},
                        'user_coverage': {'direction': 1},
                        'item_coverage': {'direction': 1},
                        'ndcg': {'direction': 1},
                        # 'blockbuster_share' : {'direction': -1}
                        }
        self.k_cf = k_cf
        self.k_mc = k_mc

    def init_model(self):
        ''' Initialize the model parameters
        '''
        self.V_user_item = self.init_sigma * np.random.randn(self.n_users, self.k_cf).astype(np.float32)
        self.V_item_user = self.init_sigma * np.random.randn(self.n_items, self.k_cf).astype(np.float32)
        self.V_prev_next = self.init_sigma * np.random.randn(self.n_items, self.k_mc).astype(np.float32)
        self.V_next_prev = self.init_sigma * np.random.randn(self.n_items, self.k_mc).astype(np.float32)

    def prepare_model(self):
        #所谓的验证集、测试集、训练集应该怎么处理？
        #疑问1：训练集应该包含测试集和训练集中所有的user和item吗？如果不包含，是不是所谓的cold-start问题？
        #疑问2：如果出现了cold-start，应该如何处理？
        #2019-8-3:按照这里的代码逻辑，不涉及cold start问题。
        '''Must be called before using train, load or top_k_recommendations
        '''
        dataset = pd.read_csv("DataSet/yelp.train.rating", sep='\t', header=None, usecols=[0, 1, 2, 3], dtype={0: np.int32, 1: np.int64, 2: np.float32, 3: str})
        dataset.columns = ["SessionId", "ItemId", "Rating", "Time"]
        dataset.sort_values(["SessionId", "Time"], inplace=True)
        dataset = dataset.reset_index(drop=True)
        self.dataset = dataset
        self.n_items = dataset.ItemId.nunique()
        self.n_users = dataset.SessionId.nunique()

        test_set = []
        val_set = []

        for userid in range(self.n_users):
            temp_df = self.dataset[self.dataset.SessionId == userid]
            temp_list = []#[userid, (item_id, last_item_id), rating]
            last_item_id = temp_df[-3:-2].iloc[0, 1]
            item_id = temp_df[-2:-1].iloc[0, 1]
            temp_list.append(temp_df[-2:-1].iloc[0, 0])
            temp_list.append((item_id, last_item_id))
            temp_list.append(temp_df[-2:-1].iloc[0, 2])
            val_set.append(temp_list)

        for userid in range(self.n_users):
            temp_df = self.dataset[self.dataset.SessionId == userid]
            temp_list = []  # [userid, (item_id, last_item_id), rating]
            last_item_id = temp_df[-2:-1].iloc[0, 1]
            item_id = temp_df[-1:].iloc[0, 1]
            temp_list.append(temp_df[-1:].iloc[0, 0])
            temp_list.append((item_id, last_item_id))
            temp_list.append(temp_df[-1:].iloc[0, 2])
            test_set.append(temp_list)

        list_train = []

        for userid in range(self.n_users):
            list_train.append(self.dataset[self.dataset.SessionId == userid][:-2])

        train_df = pd.concat(list_train, axis=0)
        train_df = train_df.reset_index(drop=True)

        self.trainset = train_df
        self.valset = val_set
        self.testset = test_set

        self.ratings_dict = {}#数据结构为dict的dict，即每个user有一个评分dict，在形成训练样本时有用。
        self.user_input, self.item_input_prev, self.item_input_next = [], [], []
        for userid in range(self.n_items):
            for row in self.dataset[self.dataset.SessionId == userid].itertuples(index=False):
                if row[0] not in self.ratings_dict:
                    self.ratings_dict[row[0]] = {}
                    self.ratings_dict[row[0]][row[1]] = row[2]
                else:
                    self.ratings_dict[row[0]][row[1]] = row[2]

        prev_item = -1
        current_item = -1
        current_user = -1
        next_user = -1
        for row in self.trainset.itertuples(index=False):
            next_user, current_item = row[0], row[1]
            if next_user != current_user:
                current_user = next_user
                prev_item = current_item
            else:
                current_user = next_user
                self.user_input.append(next_user)
                self.item_input_prev.append(prev_item)
                self.item_input_next.append(current_item)
                prev_item = current_item

    def _get_model_filename(self, epochs):
        '''Return the name of the file to save the current model
        '''
        filename = "fpmc_ne"+str(epochs)+"_lr"+str(self.init_learning_rate)+"_an"+str(self.annealing_rate)+"_kcf"+str(self.k_cf)+"_kmc"+str(self.k_mc)+"_reg"+str(self.reg)+"_ini"+str(self.init_sigma)
        return filename+".npz"

    def change_data_format(self, dataset):
        # 这里针对items主要有三个变量，并且后面也会用到
        # item_map, items, item_list, 不是很懂这三者之间的关系，并且不明白这是怎么和csr格式联系在一起的？
        '''Gets a generator of data in the sequence format and save data in the csr format
        '''
        # csr格式，好像是稀疏矩阵的一种存储方式
        #2019-8-3：这个函数应该没什么用
        self.users = np.zeros((self.n_users, 2), dtype=np.int32)
        self.items = np.zeros(len(dataset), dtype=np.int32)

        index_session = dataset.columns.get_loc('SessionId')
        index_item = dataset.columns.get_loc('ItemId')

        session_list = []

        self.user_map = {}
        self.user_count = 0

        self.item_map = {}
        self.item_list = []
        self.item_count = 0

        last_session = -1

        cursor = 0

        for row in dataset.itertuples(index=False):

            item, session = row[index_item], row[index_session]

            if not session in self.user_map:
                self.user_map[session] = self.user_count
                self.user_count += 1

            if not item in self.item_map:
                self.item_map[item] = self.item_count
                self.item_list.append(item)
                self.item_count += 1

            if last_session != session:
                muser = self.user_map[session]  # 到session为止的用户累积和

                if last_session > 0:
                    self.users[muser, :] = [cursor, len(session_list)]
                    self.items[cursor:cursor + len(session_list)] = session_list
                    cursor += len(session_list)

                session_list = []

            last_session = session
            session_list.append(self.item_map[item])

        self.users[muser, :] = [cursor, len(session_list)]
        self.items[cursor:cursor + len(session_list)] = session_list
        cursor += len(session_list)

    def get_pareto_front(self, metrics, metrics_names):
        costs = np.zeros((len(metrics[metrics_names[0]]), len(metrics_names)))
        for i, m in enumerate(metrics_names):
            costs[:, i] = np.array(metrics[m]) * self.metrics[m]['direction']
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] >= c, axis=1)
        return np.where(is_efficient)[0].tolist()

    def _compute_validation_metrics(self, metrics):
        ev = evaluation_val.Evaluator(self.dataset, k=10)

        for userid in range(self.n_users):
            if  self.valset[userid][2] > 1.0:#这里要考虑到自己做的假设，与BPR不同，只以在大于1.0评分上的记录的指标为标准。
                seq = [[self.valset[userid][1][1]]]
                top_k = self.top_k_recommendations(seq, user_id=userid)
                ev.add_instance([self.valset[userid][1][0]], top_k)
        # session_idx = self.valset.columns.get_loc('SessionId')
        # item_idx = self.valset.columns.get_loc('ItemId')
        #
        # last_item = -1
        # last_session = -1
        # for row in self.valset.itertuples(index=False):
        #     item, session = row[item_idx], row[session_idx]
        #
        #     if last_session != session:
        #         last_item = -1
        #
        #     if last_item != -1:
        #         seq = [[last_item]]
        #         top_k = self.top_k_recommendations(seq, user_id=session)
        #         ev.add_instance([item], top_k)
        #         # seq = [[self.item_map[last_item]]]
        #         #top_k = self.top_k_recommendations(seq, user_id=self.user_map[session])
        #         #ev.add_instance([self.item_map[item]], top_k)
        #
        #     last_item = item
        #     last_session = session

        metrics['recall'].append(ev.average_recall())
        metrics['sps'].append(ev.sps())
        metrics['ndcg'].append(ev.average_ndcg())
        metrics['user_coverage'].append(ev.user_coverage())
        metrics['item_coverage'].append(ev.item_coverage())

        return metrics

    def train(self, dataset,
                max_time=np.inf,
                progress=100000,
                time_based_progress=False,
                autosave='Best',
                save_dir='mdl/',
                min_iter=100000,
                max_iter=2000000,
                max_progress_interval=np.inf,
                load_last_model=False,
                early_stopping=None,
                validation_metrics=['sps']):
            # 转换数据格式，初始化相关变量（在adapter.py中已经初始化了）
            #self.prepare_model(dataset)
            # del dataset.training_set.lines
            # if len(set(validation_metrics) & set(self.metrics.keys())) < len(validation_metrics):
            #     raise ValueError(
            #         'Incorrect validation metrics. Metrics must be chosen among: ' + ', '.join(self.metrics.keys()))

            # Load last model if needed, else initialise the model
            iterations = 0
            epochs_offset = 0
            if load_last_model:
                epochs_offset = self.load_last(save_dir)
            if epochs_offset == 0:
                self.init_model()
                # python 是动态语言，是在运行期执行，而不是在编译期执行，所以不管定义的方法是在父类还是子类，只要该对象有该方法就可以了，表面
            # 表面看起来就像父类调用子类的方法一样，链接：https://www.cnblogs.com/jianyungsun/p/6288047.html

            start_time = time()
            next_save = int(progress)
            train_costs = []  # 每min_iterations迭代后，对该min_iteration迭代内的所有error求个均值，在添加进来
            current_train_cost = []  # 记录一次sgd训练更新得到的误差，该List最大有min_iteration个sgd
            # 更新误差，达到min_iteration后，立即清空
            epochs = []
            metrics = {name: [] for name in self.metrics.keys()}
            filename = {}

            while (time() - start_time < max_time and iterations < max_iter):
                # 根据目前的理解，这里的收敛停止准则是认为给定一个最大迭代次数？或许是无法达到完美的收敛准则？
                # 应该就是人为设置一个最大迭代次数作为停止条件
                cost = self.training_step(iterations)  # 即一步SGD更新得到的ERROR

                current_train_cost.append(cost)

                # Cool learning rate   为什么不是直接设置为初始的学习速率1？
                if iterations % len(dataset) == 0:
                    self.learning_rate *= self.annealing_rate

                # Check if it is time to save the model
                iterations += 1

                if time_based_progress:
                    progress_indicator = int(time() - start_time)
                else:
                    progress_indicator = iterations

                if progress_indicator >= next_save:

                    if progress_indicator >= min_iter:

                        # Save current epoch
                        epochs.append(epochs_offset + iterations / len(dataset))

                        # Average train cost
                        train_costs.append(np.mean(current_train_cost))  # min_iterations的迭代次数的平均训练误差
                        current_train_cost = []  # min_iterations迭代次数后将这个list容器重置

                        # Compute validation cost 在min_iterations迭代次数后计算在验证集上的各种评价指标，
                        # 返回的为dict，关键字为各个评价指标，键值为相应的值
                        metrics = self._compute_validation_metrics(metrics)

                        # Print info
                        self._print_progress(iterations, epochs[-1], start_time, train_costs, metrics,
                                             validation_metrics)

                        # Save model
                        # 这一块真的不是很懂
                        # 这一块代码的作用理解：
                        # 因为要在验证集上计算大概epochs次的评估指标计算，每一次min_iterations完后
                        # 我们就会得到一个模型的参数，而没得到一次参数之后，程序需要将相应的参数保存到对应的文件里
                        # 由于这里是人为给定最大迭代次数作为收敛条件，所以越到后面的参数肯定是最好的
                        # 所以如果选择“best”参数，那么就需要在每一次得到新的参数文件之后，删除掉旧之前保存的旧的参数
                        # 文件，如果autosave = all，那么就简单很多了，直接保存每一次的参数文件就ok了。
                        #                    '''
                        #                    这里的代码片段可以这样理解：
                        #                    保留在验证集中评估效果最好的参数文件，如果最开始的参数是最好的，
                        #                    那么只会有一个保留文件的操作，如果最开始的参数在验证集上的表现不是最好的，
                        #                    那么会出现保留文件和删除文件的操作
                        #                    '''
                        run_nb = len(metrics[list(self.metrics.keys())[0]]) - 1  # 这里-1是为了和pareto_runs对应
                        # 因为这里下标也是从0开始的，这样下标的计数方式保持一致
                        # 因为要进行epochs次的数据集扫描，所以metrics中的某一个尺度对应的结果就有多个，
                        if autosave == 'All':
                            filename[run_nb] = save_dir + self._get_model_filename(round(epochs[-1], 3))
                            self.save(filename[run_nb])
                        elif autosave == 'Best':
                            pareto_runs = self.get_pareto_front(metrics, validation_metrics)
                            # 好像返回的是到目前的迭代轮次为止，一共在验证集上进行了多少次的验证计算，返回的是单元素的List
                            if run_nb in pareto_runs:
                                filename[run_nb] = save_dir + self._get_model_filename(round(epochs[-1], 3))
                                self.save(filename[run_nb])
                                to_delete = [r for r in filename if r not in pareto_runs]
                                for run in to_delete:
                                    try:
                                        os.remove(filename[run])  # 这里的删除是将相应的文件删除掉
                                        print('Deleted ', filename[run])
                                    except OSError:
                                        print('Warning : Previous model could not be deleted')
                                    del filename[run]  # 这里的删除是将这个key_values对从filename字典中删除

                        if early_stopping is not None:
                            # Stop if early stopping is triggered for all the validation metrics
                            if all([early_stopping(epochs, metrics[m]) for m in validation_metrics]):
                                break

                            # Compute next checkpoint
                    if isinstance(progress, int):
                        next_save += min(progress, max_progress_interval)
                    else:
                        next_save += min(max_progress_interval, next_save * (progress - 1))

            best_run = np.argmax(
                np.array(metrics[validation_metrics[0]]) * self.metrics[validation_metrics[0]]['direction'])
            # 这里不是很懂？或者自己代入变量的时候有问题？
            return ({m: metrics[m][best_run] for m in self.metrics.keys()}, time() - start_time, filename[best_run])
        # 返回的是一个tuple，tuple的第一个元素记录了在验证上计算得到的最好结果，是一个dict，记录了每种尺度的最好结果

    def _print_progress(self, iterations, epochs, start_time, train_costs, metrics, validation_metrics):
        '''Print learning progress in terminal
        #这个函数的含义基本搞懂
        '''
        print(self.name, iterations, "batchs, ", epochs, " epochs in", time() - start_time, "s")
        print("Last train cost : ", train_costs[-1])
        for m in self.metrics:
            print(m, ': ', metrics[m][-1])
            if m in validation_metrics:
                print('Best ', m, ': ', max(np.array(metrics[m])*self.metrics[m]['direction'])*self.metrics[m]['direction'])
        print('-----------------')

    #仿佛记得这个函数没用？
    def load_last(self, save_dir):
        '''Load last model from dir
        '''

        def extract_number_of_epochs(filename):
            m = re.search('_ne([0-9]+(\.[0-9]+)?)_', filename)
            return float(m.group(1))

        # Get all the models for this RNN
        file = save_dir + self._get_model_filename("*")
        file = np.array(glob.glob(file))

        if len(file) == 0:
            print('No previous model, starting from scratch')
            return 0

        # Find last model and load it
        last_batch = np.amax(np.array(map(extract_number_of_epochs, file)))
        last_model = save_dir + self._get_model_filename(last_batch)
        print('Starting from model ' + last_model)
        self.load(last_model)

        return last_batch

    def training_step(self, iterations):
        return self.sgd_step(*self.get_training_sample())

    def sgd_step(self, user, prev_item, true_next, false_next):
        ''' Make one SGD update, given that the transition from prev_item to true_next exist in user history,
        But the transition prev_item to false_next does not exist.
        user, prev_item, true_next and false_next are all user or item ids.

        return error
        这里考虑的basket的size为1
        '''

        # Compute error
        x_true = np.dot(self.V_user_item[user, :], self.V_item_user[true_next, :]) + np.dot(self.V_prev_next[prev_item, :], self.V_next_prev[true_next, :])
        x_false = np.dot(self.V_user_item[user, :], self.V_item_user[false_next, :]) + np.dot(self.V_prev_next[prev_item, :], self.V_next_prev[false_next, :])
        delta = 1 - 1 / (1 + math.exp(min(10, max(-10, x_false - x_true)))) # Bound x_true - x_false in [-10, 10] to avoid overflow

        # Update CF MF的更新因子
        V_user_item_mem = self.V_user_item[user, :]
        self.V_user_item[user, :] += self.learning_rate * ( delta * (self.V_item_user[true_next, :] - self.V_item_user[false_next, :]) - self.reg * self.V_user_item[user, :])
        self.V_item_user[true_next, :] += self.learning_rate * ( delta * V_user_item_mem - self.reg * self.V_item_user[true_next, :])
        self.V_item_user[false_next, :] += self.learning_rate * ( -delta * V_user_item_mem - self.reg * self.V_item_user[false_next, :])

        # Update MC MC的更新因子
        V_prev_next_mem = self.V_prev_next[prev_item, :]
        self.V_prev_next[prev_item, :] += self.learning_rate * ( delta * (self.V_next_prev[true_next, :] - self.V_next_prev[false_next, :]) - self.reg * self.V_prev_next[prev_item, :])
        self.V_next_prev[true_next, :] += self.learning_rate * ( delta * V_prev_next_mem - self.reg * self.V_next_prev[true_next, :])
        self.V_next_prev[false_next, :] += self.learning_rate * ( -delta * V_prev_next_mem - self.reg * self.V_next_prev[false_next, :])

        return delta

    def get_training_sample(self):
        '''Pick a random triplet from self.triplets and a random false next item.
        returns a tuple of ids : (user, prev_item, true_next, false_next)
        '''
        #这里在挑选一个训练样本的时候，不是按照implicit feedback下的BPR假设来；
        #和rating相关联，如果评分为1，那么将true next和false next分开来。
        index = random.randrange(len(self.user_input))
        user_id = self.user_input[index]
        item_prev = self.item_input_prev[index]
        item_true_next = self.item_input_next[index]
        item_true_next_rating = self.ratings_dict[user_id][item_true_next]
        item_false_next = random.randrange (self.n_items)
        while item_false_next == item_true_next:
            item_false_next = random.randrange(self.n_items)
        if item_true_next_rating < 2.0:
            return (user_id, item_prev, item_false_next, item_true_next)
        else:
            return (user_id, item_prev, item_true_next, item_false_next)
        # while self.users[user_id,1] < 2:
        #     user_id = random.randrange(self.n_users)
        # r = random.randrange(self.users[user_id,1]-1)
        # prev_item = self.items[self.users[user_id,0]+r]
        # true_next = self.items[self.users[user_id,0]+r+1]
        # #按照这份代码的话，进行sgd更新的时候用到的是在训练集中某个用户最新的两次相邻记录
        # false_next = random.randrange(self.n_items-1)
        # if false_next >= true_next: # To make sure false_next != true_next 这一步操作不是很明白？
        #         #如果只是为了达到不等于的目的话，那直接改成！=不就完了吗
        #     false_next += 1
        #
        # return (user_id, prev_item, true_next, false_next)

    def top_k_recommendations(self, sequence, user_id=None, k=10, exclude=None, session=None):
        ''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
        这里难道是代码作者注释错了？sequence的格式不对啊
        '''
        #测试集中测试时传入user_id的原因：
        #因为整个模型是在train set上训练的，而valset是train set的一部分，是属于包含的关系
        #所以传user_id，不会出现keyerror
        #但是train set和test set却不是包含关系，test set中的user_id和train set中的user_id不同
        #那么在测试集中用训练好的模型进行评分计算时，若传入user_id，则会出现keyerror的情况
        #2019-7-20：在别的论文中看到，这种操作就属于冷启动问题了，给你一个新用户，怎么去操作？一般
        #那些没有考虑冷启动问题的实验，测试集中的用户和item都是包含在训练集中的。
        #在valset进行验证时，无论是源代码还是自己的代码，都不存在cold-start的问题。
        if exclude is None:
            exclude = []

        last_item = sequence[-1][0]
        if user_id is None:
            uv = self.V_item_user[session].mean(axis=0)
            #这一步，虽然能保证最后的结果的形式是可以计算的，可是为什么能这么计算？
            #2019-8-3：可能是在testset上针对cold start的计算方案。
        else:
            uv = self.V_user_item[user_id, :]
        output = np.dot(uv, self.V_item_user.T) + np.dot(self.V_prev_next[last_item, :], self.V_next_prev.T)

        # Put low similarity to viewed items to exclude them from recommendations
        output[[i[0] for i in sequence]] = -np.inf
        output[exclude] = -np.inf

        # find top k according to output
        return list(np.argpartition(-output, range(k))[:k])

    def recommendations(self, item_id, user_id):
            ''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
            debug的时候发现传入的sequence并不是(id, rating)格式
             这里的推荐和上一个top_k推荐有什么不同吗？
            '''

            # if user_id is None:
            #     uv = self.V_item_user[session].mean(axis=0)  # 这里的求推荐得分的操作没看懂
                # 这样操作的可能原因：当测试集中出现了训练集中没有的sessionid时，通过这种方式来求解最终的推荐分数
                # 但是给的代码貌似每个sessionid都是这样做的，这就不能理解了，而且是用的
                # V_item_user来做的，这更不能理解了
            #        '''
            #        2019-6-15
            #        好像有点明白这样做的原因了：
            #        train set和test set中的user_id是完全互斥的，所以对test set中的每一个user_id只能传入session参数
            #        如果传入session参数的话，最后的评分结果的理解：
            #        计算的是items之间的相似性加上MC的转移概率模型
            #        不对，这样理解也有点问题，这个FPMC的模型里面没说可以这么计算啊？
            #        PS：V_item_user:represent the item latent factors regarding
            #        the previously examined item。从这个角度看的话，在测试集上进行评分计算的时候，
            #        评分由两部分组成，1、V_item_user和V_item_user的转置相乘；2、MC模型的得分计算
            #        但是FPMC论文中好像没有明确说明这一点
            #        '''
            # else:
            uv = self.V_user_item[user_id, :]
            output = np.dot(uv, self.V_item_user.T) + np.dot(self.V_prev_next[item_id, :],
                                                             self.V_next_prev.T)  # 计算training set中所有item的得分，

            return output

    def save(self, filename):
        '''Save the parameters of a network into a file
        '''
        print('Save model in ' + filename)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        np.savez(filename, V_user_item=self.V_user_item, V_item_user=self.V_item_user, V_prev_next=self.V_prev_next, V_next_prev=self.V_next_prev)

    def load(self, filename):
        '''Load parameters values form a file
        '''
        f = np.load(filename)
        self.V_user_item = f['V_user_item']
        self.V_item_user = f['V_item_user']
        self.V_prev_next = f['V_prev_next']
        self.V_next_prev = f['V_next_prev']