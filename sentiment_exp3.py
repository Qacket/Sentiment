import random

import pandas as pd
import scipy.stats
import numpy as np
from multiprocessing import Pool

from scipy import stats

from converge.converge_data import converging
from generate.generate_data import generating_fixed_task, generating_fixed_annotator, generating
from sample.sample_data import sampling
from utils import get_datasets_p_distribution, Pre_Kolmogorov_Smirnov

if __name__ == '__main__':
    main_dir = r'./datasets'
    crowd_file = main_dir + '/sentiment_crowd.txt'
    truth_file = main_dir + '/sentiment_truth.txt'
    generate_methods = ['DS_generate', 'HDS_generate', 'MV_generate', 'GLAD_generate', 'RDG_generate', 'Mymodel_generate']
    converge_methods = ['DS', 'HDS', 'MV', 'GLAD']
    data_scale_range = range(500, 5000, 500)
    max_sampling_time = 10
    max_generate_sum = 10
    pool = Pool()

    for number in data_scale_range:  # 采样
        for i in range(max_sampling_time):
            sample_file = main_dir + '/sentiment_3/' + str(number) + '/sampling/sampling_' + str(i) + '.txt'
            sampling(crowd_file, number).run(sample_file)
            for generate_method_name in generate_methods:  # 一个样本文件在 每个生成方法下生成数据
                for j in range(max_generate_sum):
                    generate_file = main_dir + '/sentiment_3/' + str(number) + '/' + generate_method_name + '/' + str(i) + '_' + str(j) + '.txt'
                    if generate_method_name == 'Mymodel_generate':
                        generating(crowd_file, truth_file, sample_file, generate_method_name, generate_file)  # 生成20个数据集
                    else:
                        pool.apply_async(generating,
                                         args=(crowd_file, truth_file, sample_file, generate_method_name, generate_file))

    pool.close()
    pool.join()

    # # KL
    # mean_KL_list = []
    # mean_KL_path = main_dir + '/sentiment_3/KL_mean.csv'
    # std_KL_list = []
    # std_KL_path = main_dir + '/sentiment_3/KL_std.csv'
    #
    # for number in data_scale_range:
    #     for generate_method_name in generate_methods:  # 生成方法 5
    #         KL_list = []
    #         KL_path = main_dir + '/sentiment_3/' + str(number) + '/' + generate_method_name + '/KL_total.csv'
    #         for i in range(max_sampling_time):
    #             sample_path = main_dir + '/sentiment_3/' + str(number) + '/sampling/sampling_' + str(i) + '.txt'
    #             sample_probability_distribution, sample_p = get_datasets_p_distribution(sample_path)
    #             for j in range(max_generate_sum):
    #                 generate_file = main_dir + '/sentiment_3/' + str(number) + '/' + generate_method_name + '/' + str(
    #                     i) + '_' + str(j) + '.txt'
    #                 generate_probability_distribution, generate_p = get_datasets_p_distribution(generate_file)
    #                 KL = scipy.stats.entropy(sample_p, generate_p)
    #                 KL_list.append(KL)
    #         KL_list = np.array(KL_list).reshape(-1, 1)
    #         KL_pd = pd.DataFrame(data=KL_list)
    #         KL_pd.to_csv(KL_path, index=False)
    #
    #         mean_KL = np.mean(KL_list)
    #         std_KL = np.std(KL_list, ddof=1)
    #
    #         mean_KL_list.append(mean_KL)
    #         std_KL_list.append(std_KL)
    #
    # mean_KL_np = np.array(mean_KL_list).reshape(-1, 6)
    # mean_KL_pd = pd.DataFrame(data=mean_KL_np)
    # mean_KL_pd.to_csv(mean_KL_path, index=False)
    #
    # std_KL_np = np.array(std_KL_list).reshape(-1, 6)
    # std_KL_pd = pd.DataFrame(data=std_KL_np)
    # std_KL_pd.to_csv(std_KL_path, index=False)

    # KS
    # original_data_list = Pre_Kolmogorov_Smirnov(crowd_file, truth_file)
    #
    # for number in data_scale_range:
    #
    #     mean_KS_stat_list = []
    #     mean_KS_stat_path = main_dir + '/sentiment_3/' + str(number) + '/KS_stat_mean.csv'
    #     std_KS_stat_list = []
    #     std_KS_stat_path = main_dir + '/sentiment_3/' + str(number) + '/KS_stat_std.csv'
    #
    #     mean_KS_pval_list = []
    #     mean_KS_pval_path = main_dir + '/sentiment_3/' + str(number) + '/KS_pval_mean.csv'
    #     std_KS_pval_list = []
    #     std_KS_pval_path = main_dir + '/sentiment_3/' + str(number) + '/KS_pval_std.csv'
    #
    #
    #     for generate_method_name in generate_methods:  # 生成方法 5
    #         KS_stat_list = []
    #         KS_stat_path = main_dir + '/sentiment_3/' + str(number) + '/' + generate_method_name + '/KS_stat_total.csv'
    #         KS_pval_list = []
    #         KS_pval_path = main_dir + '/sentiment_3/' + str(number) + '/' + generate_method_name + '/KS_pval_total.csv'
    #         for i in range(max_sampling_time):
    #             for j in range(max_generate_sum):
    #                 generate_file = main_dir + '/sentiment_3/' + str(number) + '/' + generate_method_name + '/' + str(
    #                     i) + '_' + str(j) + '.txt'
    #                 generate_data_list = Pre_Kolmogorov_Smirnov(generate_file, truth_file)
    #                 for k in range(8):
    #                     (ks_stat, pval) = stats.ks_2samp(original_data_list[k], generate_data_list[k])
    #                     KS_stat_list.append(ks_stat)
    #                     KS_pval_list.append(pval)
    #
    #         KS_stat_list = np.array(KS_stat_list).reshape(-1, 8)
    #         KS_stat_pd = pd.DataFrame(data=KS_stat_list)
    #         KS_stat_pd.to_csv(KS_stat_path, index=False)
    #         mean_KS_stat = np.mean(KS_stat_list, axis=0).tolist()
    #         std_KS_stat = np.std(KS_stat_list, axis=0, ddof=1).tolist()
    #         mean_KS_stat_list.append(mean_KS_stat)
    #         std_KS_stat_list.append(std_KS_stat)
    #
    #
    #         KS_pval_list = np.array(KS_pval_list).reshape(-1, 8)
    #         KS_pval_pd = pd.DataFrame(data=KS_pval_list)
    #         KS_pval_pd.to_csv(KS_pval_path, index=False)
    #         mean_KS_pval = np.mean(KS_pval_list, axis=0).tolist()
    #         std_KS_pval = np.std(KS_pval_list, axis=0, ddof=1).tolist()
    #         mean_KS_pval_list.append(mean_KS_pval)
    #         std_KS_pval_list.append(std_KS_pval)
    #
    #
    #
    #     mean_KS_stat_np = np.array(mean_KS_stat_list)
    #     mean_KS_stat_pd = pd.DataFrame(data=mean_KS_stat_np)
    #     mean_KS_stat_pd.to_csv(mean_KS_stat_path, index=False)
    #
    #     std_KS_stat_np = np.array(std_KS_stat_list)
    #     std_KS_stat_pd = pd.DataFrame(data=std_KS_stat_np)
    #     std_KS_stat_pd.to_csv(std_KS_stat_path, index=False)
    #
    #     mean_KS_pval_np = np.array(mean_KS_pval_list)
    #     mean_KS_pval_pd = pd.DataFrame(data=mean_KS_pval_np)
    #     mean_KS_pval_pd.to_csv(mean_KS_pval_path, index=False)
    #     std_KS_pval_np = np.array(std_KS_pval_list)
    #     std_KS_pval_pd = pd.DataFrame(data=std_KS_pval_np)
    #     std_KS_pval_pd.to_csv(std_KS_pval_path, index=False)

    # 汇聚
    # for number in data_scale_range:
    #     # 生成数据汇聚
    #     for generate_method_name in generate_methods:   # 生成方法 5
    #         acc_path = main_dir + '/sentiment_3/' + str(number) + '/' + generate_method_name + '/accuracy.csv'
    #         acc_applyResult_list = []
    #         for i in range(max_sampling_time):
    #             for j in range(max_generate_sum):
    #                 for converge_method_name in converge_methods:  # 汇聚方法 4
    #                     generate_file = main_dir + '/sentiment_3/' + str(number) + '/' + generate_method_name + '/' + str(i) + '_' + str(j) + '.txt'
    #                     acc = pool.apply_async(converging, args=(generate_file, truth_file, converge_method_name))
    #                     acc_applyResult_list.append(acc)
    #         acc_list = []
    #         for acc_applyResult in acc_applyResult_list:
    #             acc_list.append(acc_applyResult.get())
    #
    #         acc_list = np.array(acc_list).reshape(-1, 4)
    #         acc_pd = pd.DataFrame(data=acc_list)
    #         acc_pd.to_csv(acc_path, index=False)
    #
    # pool.close()
    # pool.join()


    # for number in data_scale_range:
    #     # 样本数据汇聚
    #     for generate_method_name in generate_methods:  # 生成方法 5
    #         acc_path = main_dir + '/sentiment_3/' + str(number) + '/sampling' + '/accuracy.csv'
    #         acc_applyResult_list = []
    #         for i in range(max_sampling_time):
    #             for converge_method_name in converge_methods:  # 汇聚方法 4
    #                 sample_file = main_dir + '/sentiment_3/' + str(number) + '/sampling/sampling_' + str(i) + '.txt'
    #                 acc = pool.apply_async(converging, args=(sample_file, truth_file, converge_method_name))
    #                 acc_applyResult_list.append(acc)
    #         acc_list = []
    #         for acc_applyResult in acc_applyResult_list:
    #             acc_list.append(acc_applyResult.get())
    #
    #         acc_list = np.array(acc_list).reshape(-1, 4)
    #         acc_pd = pd.DataFrame(data=acc_list)
    #         acc_pd.to_csv(acc_path, index=False)
    #
    # pool.close()
    # pool.join()

    # # 汇聚排序
    # for number in data_scale_range:   # 汇聚排序
    #
    #     mean_total_path = main_dir + '/sentiment_3/' + str(number) + '/mean.csv'
    #     mean_total_list = []
    #
    #     std_total_path = main_dir + '/sentiment_3/' + str(number) + '/std.csv'
    #     std_total_list = []
    #
    #     sampling_acc_path = main_dir + '/sentiment_3/' + str(number) + '/sampling' + '/accuracy.csv'
    #     sampling_acc_total = pd.read_csv(sampling_acc_path)
    #     sampling_acc_mean = sampling_acc_total.mean(axis=0)
    #     sampling_acc_std = sampling_acc_total.std(axis=0)
    #
    #     sampling_acc_mean_list = list(sampling_acc_mean)
    #     sampling_acc_std_list = list(sampling_acc_std)
    #
    #     mean_total_list = mean_total_list + sampling_acc_mean_list
    #     std_total_list = std_total_list + sampling_acc_std_list
    #
    #     for generate_method_name in generate_methods:  # 生成方法 5
    #         generate_acc_path = main_dir + '/sentiment_3/' + str(number) + '/' + generate_method_name + '/accuracy.csv'
    #         generate_acc_total = pd.read_csv(generate_acc_path)
    #         generate_acc_mean = generate_acc_total.mean(axis=0)
    #         generate_acc_std = generate_acc_total.std(axis=0)
    #
    #         generate_acc_mean_list = list(generate_acc_mean)
    #         generate_acc_std_list = list(generate_acc_std)
    #
    #         mean_total_list = mean_total_list + generate_acc_mean_list
    #         std_total_list = std_total_list + generate_acc_std_list
    #
    #     mean_total_list = np.array(mean_total_list).reshape(-1, 4)
    #     mean_total_pd = pd.DataFrame(data=mean_total_list)
    #     mean_total_pd.to_csv(mean_total_path, index=False)
    #
    #     std_total_list = np.array(std_total_list).reshape(-1, 4)
    #     std_total_pd = pd.DataFrame(data=std_total_list)
    #     std_total_pd.to_csv(std_total_path, index=False)