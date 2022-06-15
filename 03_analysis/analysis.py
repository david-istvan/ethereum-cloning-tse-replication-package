import argparse
import os
import pickle
import re
import shutil
from bdb import set_trace
from hashlib import sha256
from pprint import pformat
from statistics import mean, median, stdev
from typing import Collection

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker
from numpy import sqrt, std
from scipy import stats

from cliffsDelta import cliffsDelta

dataPath = '../01_data'
resultsPath = '../04_results'
corpusLOC = 4004543

class Analysis():

    def __init__(self):
        if not os.path.exists(resultsPath):
            os.mkdir(resultsPath)
            
    def observation1(self):
        data = pd.read_pickle(f'{dataPath}/prepared/data_rq1.p')
        
        dfg = data.groupby(['type'])['sumlines'].sum().reset_index(name ='sumlines')
        totalCloneNumber = dfg['sumlines'].sum()
        totalClonePercentage = round((totalCloneNumber*100)/corpusLOC, 2)
        report = [
            'Ratio of clones in the corpus: {}%.'.format(totalClonePercentage)
        ]
        self.printTextReport('01', report)
        
        ### Chart ###
        groups = ['clone-free', 'cloned']
        values = [corpusLOC-totalCloneNumber, totalCloneNumber]
        
        colors = ['#1f77b4', '#ff0000']

        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect='equal'))

        wedges, texts = ax.pie(values, wedgeprops=dict(width=0.48), startangle=-40, colors=colors)

        bbox_props = dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle='-'), bbox=bbox_props, zorder=0, va='center')
        
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: 'right', 1: 'left'}[int(np.sign(x))]
            connectionstyle = 'angle,angleA=0,angleB={}'.format(ang)
            kw['arrowprops'].update({'connectionstyle': connectionstyle})

        ax.legend(['{} ({}%)'.format(groups[i], round(((values[i]/corpusLOC)*100), 2)) for i in range(0,2)], loc='center', frameon=False)
        
        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        
        self.savefig('01')

    def observation2(self):
        data = pd.read_pickle(f'{dataPath}/prepared/clonesWithAuthors.p')
        
        data = data.drop_duplicates(['type', 'classid'])[['type', 'classid', 'nclones']].sort_values(by='nclones', ascending=False)
        data = data.reset_index(drop=True)
        
        sumClusters = len(data)
        sumClones = data['nclones'].sum()    
        
        data['cumulativeClusterPercentage'] = data.apply(lambda row: round(((row.name+1)/sumClusters)*100, 4), axis=1)
        data['cumulativeClonePercentage'] = round((data.nclones.cumsum()/sumClones)*100, 4)
        cumulativeClonePercentageAt2 = data.iloc[(data['cumulativeClusterPercentage']-2.07).abs().argsort()[:1]]['cumulativeClonePercentage'].values[0]
        cumulativeClonePercentageAt20 = data.iloc[(data['cumulativeClusterPercentage']-20).abs().argsort()[:1]]['cumulativeClonePercentage'].values[0]

        report = [
            'Number of clusters: {}.'.format(sumClusters),
            'Sum of clones: {}.'.format(sumClones),
            'Cumulative clone percentage at 2.07% cumulative cluster percentage: {}'.format(cumulativeClonePercentageAt2),
            'Cumulative clone percentage at 20% cumulative cluster percentage: {}'.format(cumulativeClonePercentageAt20)
        ]
        self.printTextReport('02', report)
        
        ### Chart ###
        
        plt.figure()
        x = data['cumulativeClusterPercentage']
        y = data['cumulativeClonePercentage']
        plt.plot(x,y)
        plt.xlabel('Cumulative % of clusters', fontsize=20)
        plt.ylabel('Cumulative % of clones', fontsize=20)
        
        plt.yticks(list([0, 10, 20, 30, 40, 50, 60, 80, 90, 100]) + [cumulativeClonePercentageAt20])
        plt.ylim([-5, 105])
        plt.xticks(list([0, 2.07, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
        plt.xlim([-5, 105])
        
        ax = plt.gca()
        
        labels=ax.get_xticklabels()+ax.get_yticklabels()
        for label in labels:
            label.set_fontsize(14)
        
        plt.vlines(x=2.07, ymin=-5, ymax=cumulativeClonePercentageAt2, color='red', zorder=2, linestyle='--')
        plt.hlines(y=cumulativeClonePercentageAt2, xmin=-5, xmax=2.07, color='red', zorder=2, linestyle='--')
        plt.gca().get_xticklabels()[1].set_color('r')
        plt.gca().get_yticklabels()[5].set_color('r')
        
        plt.vlines(x=20, ymin=-5, ymax=cumulativeClonePercentageAt20, color='red', zorder=2, linestyle='--')
        plt.hlines(y=cumulativeClonePercentageAt20, xmin=-5, xmax=20, color='red', zorder=2, linestyle='--')
        plt.gca().get_xticklabels()[3].set_color('red')
        plt.gca().get_yticklabels()[10].set_color('red')
        
        
        plt.grid(axis='both')
        
        plt.rcParams.update({'font.size': 12})
        
        figure = plt.gcf()
        
        figure.set_size_inches(8, 4)
        
        self.savefig('02')

    def shortenQuarter(self, quarter):
        year, q = quarter.split('.')
        return '{}/{}'.format(year.replace('20', ''), q)
        
    def observation3(self):
        data = pd.read_pickle(f'{dataPath}/prepared/data_observation3.p')
        
        quarterlyClones = data.groupby(['quarter'])[['quarter', 't1', 't2', 't2c', 't3', 't32', 't32c', 'filelength']].sum().reset_index()
        quarterlyClones = quarterlyClones.rename(columns={'t1':'type-1', 't2':'type-2b', 't2c':'type-2c', 't3':'type-3', 't32':'type-3b', 't32c':'type-3c'})
        
        quarterlyClones['t1plus'] = quarterlyClones.apply(lambda row: row['type-2b']+row['type-2c']+row['type-3']+row['type-3b']+row['type-3c'], axis=1)
        quarterlyClones['t1t3plus'] = quarterlyClones.apply(lambda row: row['type-2b']+row['type-2c']+row['type-3b']+row['type-3c'], axis=1)
        quarterlyClones['all'] = quarterlyClones.apply(lambda row: row['type-1']+row.t1plus, axis=1)
        
        quarterlyClones['quarter'] = quarterlyClones.apply(lambda row: self.shortenQuarter(row['quarter']), axis=1)
        
        self.observation3a(quarterlyClones)
        plt.clf()
        self.observation3b(quarterlyClones)
        plt.clf()
        self.observation3c(quarterlyClones)
        
    def observation3a(self, quarterlyClones):
        report = [
            ('Quarterly clones', quarterlyClones, '')
        ]
        self.printHtmlReport('03', report)
        
        ### Chart ###
        qlabels = quarterlyClones['quarter']
        x = np.arange(len(qlabels))
        width = 0.4
        
        fig, axs = plt.subplots(2)
        
        axs[0] = quarterlyClones[['all']].plot(kind='bar', ax=axs[0], width=width, rot=0, color='#777777')
        axs[0].set_ylabel('Number of new clones', fontsize=15)
        axs[0].set_ylim([0, 30000])
        axs[0].bar_label(axs[0].containers[0], fontsize=14)
        axs[0].legend(loc='upper left', fontsize=14)
        
        colors = ['#aaaaaa', '#555555']
        quarterlyT1plusClones100 = quarterlyClones[['type-3', 't1t3plus', 'all']].apply(lambda x: round(x*100/x['all'], 0), axis=1)
        quarterlyT1plusClones100 = quarterlyT1plusClones100[['type-3', 't1t3plus']]
        axs[1] = quarterlyT1plusClones100.plot(kind='bar', stacked = True, ax=axs[1], width=width, rot=0, color=colors)
        axs[1].set_ylabel('% of new non-Type-1 clones', fontsize=15)
        axs[1].set_xlabel('Quarter', fontsize=20)
        axs[1].legend(('type-3', 'other'), loc='upper left', fontsize=14)
        
        container = axs[1].containers[0]
        labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in container]
        axs[1].bar_label(container, labels=labels, label_type='center', color='black', fontsize=14)

        for ax in axs:
            ax.set_xticklabels(qlabels)
            ax.grid(axis='y')
            
            labels=ax.get_xticklabels()+ax.get_yticklabels()
            for label in labels:
                label.set_fontsize(13)

        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        
        self.savefig('03')
    
    def observation3b(self, quarterlyClones):
        ### Chart ###
        qlabels = quarterlyClones['quarter']
        x = np.arange(len(qlabels))
        width = 0.4
        
        plt.figure()
        
        ax = quarterlyClones[['filelength']].plot(kind='bar', width=width, rot=0, color='#777777')
        
        labels = list(map(lambda n: '{}M'.format(round(n/1000000, 1)) if n>1000000 else ('{}k'.format(round(n/1000, 1)) if n>1000 else n), quarterlyClones['filelength']))
        
        ax.set_ylabel('Number of new cloned lines of code (log scale)', fontsize=15)
        ax.bar_label(ax.containers[0], labels=labels, fontsize=12)
        ax.get_legend().remove()
        ax.set_yscale('log')

        ax.set_xticklabels(qlabels)
        ax.grid(axis='y')
        
        labels=ax.get_xticklabels()+ax.get_yticklabels()
        for label in labels:
            label.set_fontsize(13)

        
        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        
        self.savefig('03b')
        
    def observation3c(self, quarterlyClones):
        report = [
            ('Quarterly clones', quarterlyClones, '')
        ]
        self.printHtmlReport('03', report)
        
        ### Chart ###
        qlabels = quarterlyClones['quarter']
        x = np.arange(len(qlabels))
        width = 0.4
        
        fig, axs = plt.subplots(3)
        
        axs[0] = quarterlyClones[['all']].plot(kind='bar', ax=axs[0], width=width, rot=0, color='#777777')
        axs[0].set_ylabel('Number of new clones', fontsize=15)
        axs[0].set_ylim([0, 30000])
        axs[0].bar_label(axs[0].containers[0], fontsize=14)
        axs[0].legend(loc='upper left', fontsize=14)
        
        colors = ['#aaaaaa', '#555555']
        quarterlyT1plusClones100 = quarterlyClones[['type-3', 't1t3plus', 'all']].apply(lambda x: round(x*100/x['all'], 0), axis=1)
        quarterlyT1plusClones100 = quarterlyT1plusClones100[['type-3', 't1t3plus']]
        axs[1] = quarterlyT1plusClones100.plot(kind='bar', stacked = True, ax=axs[1], width=width, rot=0, color=colors)
        axs[1].set_ylabel('% of new non-Type-1 clones', fontsize=15)
        axs[1].legend(('type-3', 'other'), loc='upper left', fontsize=14)
        
        container = axs[1].containers[0]
        labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in container]
        axs[1].bar_label(container, labels=labels, label_type='center', color='black', fontsize=14)
        
        axs[2] = quarterlyClones[['filelength']].plot(kind='bar', ax=axs[2], width=width, rot=0, color='#777777')
        axs[2].set_ylabel('New cloned LOC (log scale)', fontsize=14, labelpad=23)
        labels = list(map(lambda n: '{}M'.format(round(n/1000000, 1)) if n>1000000 else ('{}k'.format(round(n/1000, 1)) if n>1000 else n), quarterlyClones['filelength']))
        axs[2].bar_label(axs[2].containers[0], labels=labels, fontsize=12)
        axs[2].get_legend().remove()
        axs[2].set_yscale('log')
        
        axs[2].set_xlabel('Quarter', fontsize=20)
        
        for ax in axs:
            ax.set_xticklabels(qlabels)
            ax.grid(axis='y')
            
            labels=ax.get_xticklabels()+ax.get_yticklabels()
            for label in labels:
                label.set_fontsize(13)

        figure = plt.gcf()
        figure.set_size_inches(8, 9)
        
        self.savefig('03c')
        
    def observation4(self):
        self.observation4a()
        plt.clf()
        self.observation4b()
        
    def observation4a(self):
        data = pd.read_pickle(f'{dataPath}/prepared/data_rq1.p')
        
        data = data.groupby(['type'])['sumlines'].sum().reset_index(name ='sumlines')
        totalCloneNumber = data['sumlines'].sum()
        totalClonePercentage = round((totalCloneNumber*100)/corpusLOC, 2)
        
        dfg = data
        dfg = data.append({'type' : 'clone-free', 'sumlines' : corpusLOC - dfg['sumlines'].sum()}, ignore_index=True)
        dfg['ratio'] = dfg.apply(lambda row: round((row['sumlines']*100)/corpusLOC, 2), axis=1)
        dfg = dfg.sort_values(by = 'sumlines', ascending = False)
        
        report = [
            ('Clone proportions', dfg, '')
        ]
        self.printHtmlReport('04a', report)
        
        
        ### Chart ###
        groups = ['clone-free', 'type-1', 'type-3', 'other']
        values = [corpusLOC-totalCloneNumber, data[data['type']=='type-1']['sumlines'].values[0], data.loc[data['type']=='type-3']['sumlines'].values[0], data.loc[~data.type.isin(groups)].sum().values[1]]
        percValues = [round((x/corpusLOC)*100, 2) for x in values]
        
        colors = ['#1f77b4', '#fcba03', '#e6194B', '#03fcad'] #'#911eb4', '#ffe119',  '#469990', '#42d4f4'
        
        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect='equal'))

        wedges, texts = ax.pie(values, wedgeprops=dict(width=0.48), startangle=-40, colors=colors)

        bbox_props = dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle='-'), bbox=bbox_props, zorder=0, va='center')
        
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: 'right', 1: 'left'}[int(np.sign(x))]
            connectionstyle = 'angle,angleA=0,angleB={}'.format(ang)
            kw['arrowprops'].update({'connectionstyle': connectionstyle})

        ax.legend(['{} ({}%)'.format(groups[i], round(((values[i]/corpusLOC)*100), 2)) for i in range(0,4)], loc='center', frameon=False)
        
        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        
        self.savefig('04a')
        
    def observation4b(self):
        data = pd.read_pickle(f'{dataPath}/prepared/clonesWithAuthors.p')

        data['type'] = data['type'].replace(to_replace='type-2', value='type-2b')
        data['type'] = data['type'].replace(to_replace='type-3-2', value='type-3b')
        data['type'] = data['type'].replace(to_replace='type-3-2c', value='type-3c')
        
        types = data['type'].unique()
        
        report = []
        
        report.append('Mann-Whitney between type-1 and others: {}.'.format(stats.mannwhitneyu(x=data[data['type']=='type-1']['filelength'], y=data[data['type']!='type-1']['filelength'], alternative = 'two-sided')[1]))
        es = cliffsDelta(data[data['type']=='type-1']['filelength'], data[data['type']!='type-1']['filelength'])
        if(es[1] != 'negligible'):
            report.append('\tEffect size: {} ({}).'.format(es[0], es[1]))
        
        dataFrames = []
        for type in types:
            dataFrames.append(data[data['type']==type])
        
        for i in range(0, len(dataFrames)):
            for j in range(i+1, len(dataFrames)):
                mwu = stats.mannwhitneyu(x=dataFrames[i]['filelength'], y=dataFrames[j]['filelength'], alternative = 'two-sided')
                if mwu[1] < 0.05:
                    es = cliffsDelta(dataFrames[i]['filelength'], dataFrames[j]['filelength'])
                    if(es[1] != 'negligible'):
                        report.append('Mann-Whitney between {} and {}: {}.'.format(dataFrames[i].iloc[0]['type'], dataFrames[j].iloc[0]['type'], mwu[1]))
                        report.append('\tEffect size: {} ({}).'.format(es[0], es[1]))
        
        self.printTextReport('04b', report)
        
    def observation5(self):
        #Same as observation4()
        pass
        
    def observation6(self):
        path = f'{dataPath}/clonedata/duplicates/function-ids/'
        resultsFileAll = 'all-ids.csv'
        resultsFileTop20 = 'observation06-function-ids-top20.csv'

        original_paths = ['type-1', 'type-2', 'type-2c', 'type-3-1', 'type-3-2', 'type-3-2c']

        save_path = open(f'{path}/{resultsFileAll}', 'w')
        dfs = []

        for file in original_paths:
            df = pickle.load(open(f'{path}/{file}.p', 'rb'))
            dfs.append(df)
        final_df = pd.concat(dfs, ignore_index=True)


        # THIS IS THE EXTRA NORMALIZATION STEP: which might not be necessary
        def normalize_func_signatures(id):
            pa = r'([\s\S]*?)\(([\s\S]*?)\)([\s\S]*)'
            matches = re.match(pa, id, re.MULTILINE)
            matches = [g.strip() for g in matches.groups()]

            def ident(x):
                return x.strip()
                
            def params(x):
                x = set([y.strip() for y in x.split(',')])
                return x

            def return_types(x):
                tre = re.compile(r"""[\s,]*(~@|[\[\]{}()'`~^@]|"(?:[\\].|[^\\"])*"?|;.*|[^\s\[\]{}()'"`@,;]+)""");
                tokens = [t for t in re.findall(tre, x)]

                ret_seq = []
                def read_sequence(tokens, pointer, ret_seq, seq_start='(', seq_end=')'):
                    if pointer >= len(tokens):
                        return pointer, ret_seq

                    prev = ret_seq.pop() if len(ret_seq)>0 else ''
                    while tokens[pointer] != seq_end:
                        if tokens[pointer] == seq_start:
                            prev+=seq_start
                            pointer +=1
                        else:
                            nested_read = []
                            pointer, nested_read = reader(tokens, pointer, nested_read)
                            prev += ' '.join(nested_read)

                    prev += seq_end
                    ret_seq.append(prev)
                    return reader(tokens, pointer+1, ret_seq)

                def read_atom(tokens, pointer, ret_seq):
                    if pointer >= len(tokens):
                        return pointer, ret_seq

                    if tokens[pointer] == 'public':
                        # skip public: solidity functions are public by default
                        return reader(tokens, pointer+1, ret_seq)
                    ret_seq.append(tokens[pointer])
                    return reader(tokens, pointer+1, ret_seq)

                def reader(tokens, pointer, ret_seq):
                    if pointer >= len(tokens):
                        return pointer, ret_seq
                    elif tokens[pointer] == '(':
                        pointer, ret_seq = read_sequence(tokens, pointer, ret_seq, '(', ')')
                    elif tokens[pointer] == '[':
                        pointer, ret_seq = read_sequence(tokens, pointer, ret_seq, '[', ']')
                    elif tokens[pointer] == ')':
                        return pointer, ret_seq
                    elif tokens[pointer] == ']':
                        return pointer, ret_seq
                    else:
                        pointer, ret_seq = read_atom(tokens, pointer, ret_seq)
                    return pointer, ret_seq

                pointer, ret_seq = reader(tokens, 0, ret_seq)
                return set(ret_seq)

            return f'{ident(matches[0])}({str(params(matches[1]))}) {str(return_types(matches[2]))}'

        new_ids = final_df.ids.apply(normalize_func_signatures)
        # SKIP THIS LINE TO REMOVE FUNCTION SIGNATURE NORMALIZATION
        final_df['ids'] = new_ids.apply(lambda x:x.replace('{', '').replace('}','').replace("'", ''))

        all_functions = final_df.groupby(['ids']).count().sort_values(by='type', ascending=False).reset_index()[['ids','type']]
        all_functions.columns = ['functionID', 'count']
        pd.set_option('max_colwidth', 1000)
        report = [
            ('All functions', all_functions, '')
        ]
        self.printHtmlReport('06', report)
        
        report = [
            ('Top 20 functions', all_functions[:20], '')
        ]

        self.printHtmlReport('06b', report)

    def observation7(self):
        # Same as observation 6
        pass
    
    def observation8(self):
        data = pd.read_pickle(f'{dataPath}/prepared/gini.p')
        
        data['type'] = data['type'].replace(to_replace='type-2', value='type-2b')
        data['type'] = data['type'].replace(to_replace='type-3-2', value='type-3b')
        data['type'] = data['type'].replace(to_replace='type-3-2c', value='type-3c')
        
        data['nclonesperc'] = data.apply(lambda row: round((row['nclones']/(data['nclones'].max())),2), axis=1)
        data = data.sort_values(by=['nclonesperc'])
        
        self.observation8a(data)
        self.observation8b(data)
        self.observation8c(data)
    
    def observation8a(self, data):        
        #HEXBIN CHART
        fig = plt.figure()
        ax = plt.gca()
        
        x = data['nclonesperc']
        y = data['gini']
        hb = ax.hexbin(x, y, gridsize = 35, cmap ='binary', edgecolor='gray', mincnt=1)
        
        plt.axvline(x=round(data['nclones'].median()/data['nclones'].max(), 2), color='r')
        plt.axhline(y=data['gini'].median(), color='r')
        plt.yticks(list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, round(data['gini'].median(),2), 0.9, 1.0]))
        plt.gca().get_yticklabels()[9].set_color('red')
        ax.text(15.75, -0.1, str(int(data['nclones'].median())), color='red', fontsize=16)

        plt.xlabel('Normalized cluster size', fontsize=16)
        plt.ylabel('Gini-coefficient', fontsize=16)
        
        labels=ax.get_xticklabels()+ax.get_yticklabels()
        for label in labels:
            label.set_fontsize(16)
        
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Count', fontsize=16)
        cb.ax.tick_params(labelsize=16)
        
        fig.set_size_inches(8, 6)
        
        self.savefig('08a')
        
    def observation8b(self, data):        
        #BOX PLOT
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 6), gridspec_kw={'width_ratios': [4, 3]})
        
        bp1 = data.boxplot(column=['gini'], by='type', patch_artist = True, return_type='both', medianprops=dict(linewidth=1, color='black'), whiskerprops=dict(linewidth=1, color='black'), ax=axs[0])
        
        for row_key, (ax,row) in bp1.iteritems():
            ax.set_xlabel('')
            for i,box in enumerate(row['boxes']):
                box.set(color='#aaaaaa', linewidth=2)
        
        bp2 = data.boxplot(column=['gini'], patch_artist = True, return_type='both', boxprops=dict(facecolor='#aaaaaa', color='#aaaaaa'), medianprops=dict(linewidth=1, color='black'), whiskerprops=dict(linewidth=1, color='black'), ax=axs[1])
        
        axs[0].set_title('')
        axs[1].set_title('')
        fig.suptitle('')
        
        axs[0].set_ylabel('Gini-coefficient', fontsize=16)
        axs[1].set_xticklabels(['Overall'])
        
        for ax in axs:
            labels=ax.get_xticklabels()+ax.get_yticklabels()
            for label in labels:
                label.set_fontsize(16)

        fig.set_size_inches(8, 4)
        
        self.savefig('08b')
        
    def observation8c(self, data):
        reportDf = data[['type']]
        reportDf = reportDf.drop_duplicates().reset_index(drop=True)
        reportDf['median'] = reportDf.apply(lambda row: median(data.loc[(data['type']==row['type'])]['gini']), axis=1)
        
        reportDf = reportDf.sort_values(by=['type']).reset_index(drop=True)
        
        reportDf = reportDf.append({'type' : 'overall', 'median' : median(data['gini'])}, ignore_index = True)
        
        report = [
            ('Median Gini-coefficients', reportDf[['type', 'median']], 'G=0.85 roughly equals to a cluster of 10 contracts with nine contracts having 1 transaction, and one transaction having 250.<br/>G=0.75 roughly equals to a cluster of 10 contracts with nine contracts having 1 transaction, and one transaction having 50.<br/>')
        ]
        
        self.printHtmlReport('08c', report)
    
    def observation9(self):
        data = pd.read_pickle(f'{dataPath}/prepared/creationrank.p')
        
        data['type'] = data['type'].replace(to_replace='type-2', value='type-2b')
        data['type'] = data['type'].replace(to_replace='type-3-2', value='type-3b')
        data['type'] = data['type'].replace(to_replace='type-3-2c', value='type-3c')
        
        data = data[data['nclones'] >= 10].reset_index(drop=True)

        data['rankPercentage'] = data.apply(lambda row: round((row['topTxCreationRank']/row['nclones'])*100, 2), axis=1)
        data = data.sort_values(by='rankPercentage').reset_index(drop=True)
        
        self.observation9a(data)
        self.observation9b(data)
    
    def observation9a(self, data):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 6), gridspec_kw={'width_ratios': [4, 3]})
        
        bp1 = data.boxplot(column=['rankPercentage'], by='type', patch_artist = True, return_type='both', medianprops=dict(linewidth=1, color='black'), whiskerprops=dict(linewidth=1, color='black'), ax=axs[0])
        colors = ['#1f77b4', '#ffe119', '#e6194B', '#469990']
        
        for row_key, (ax,row) in bp1.iteritems():
            ax.set_xlabel('')
            for i,box in enumerate(row['boxes']):
                box.set(color='#aaaaaa', linewidth=2)
        
        bp2 = data.boxplot(column=['rankPercentage'], patch_artist = True, return_type='both', boxprops=dict(facecolor='#aaaaaa', color='#aaaaaa'), medianprops=dict(linewidth=1, color='black'), whiskerprops=dict(linewidth=1, color='black'), ax=axs[1])
        
        axs[0].set_title('')
        axs[1].set_title('')
        fig.suptitle('')
        
        axs[0].set_ylabel('Rank percentage', fontsize=16)
        axs[1].set_xticklabels(['Overall'])
        
        for ax in axs:
            labels=ax.get_xticklabels()+ax.get_yticklabels()
            for label in labels:
                label.set_fontsize(16)

        fig.set_size_inches(8, 4)
        
        self.savefig('09')
        
    def observation9b(self, data):
        report = [
            '[OVERALL] Median of rank percentage: {}.'.format(median(data['rankPercentage'])),
            '[OVERALL] Median of rank: {}.'.format(median(data['topTxCreationRank'])),
            '[TYPE-1] Median of rank percentage: {}.'.format(median(data[data['type']=='type-1']['rankPercentage'])),
            '[TYPE-1] Median of rank: {}.'.format(median(data[data['type']=='type-1']['topTxCreationRank'])),
            '[TYPE-2C] Median of rank percentage: {}.'.format(median(data[data['type']=='type-2c']['rankPercentage'])),
            '[TYPE-2C] Median of rank: {}.'.format(median(data[data['type']=='type-2c']['topTxCreationRank'])),
            '[TYPE-3] Median of rank percentage: {}.'.format(median(data[data['type']=='type-3']['rankPercentage'])),
            '[TYPE-3] Median of rank: {}.'.format(median(data[data['type']=='type-3']['topTxCreationRank'])),
            '[TYPE-3B] Median of rank percentage: {}.'.format(median(data[data['type']=='type-3b']['rankPercentage'])),
            '[TYPE-3B] Median of rank: {}.'.format(median(data[data['type']=='type-3b']['topTxCreationRank'])),
        ]
        self.printTextReport('09', report)
        
    def observation10(self):
        authorData = pd.read_pickle(f'{dataPath}/prepared/data_rq2.p')
        
        self.observation10a(authorData)
        plt.clf()
        self.observation10b(authorData)
    
    def observation10a(self, authorData):
        authorData = authorData.sort_values(by=['entropy'], ascending=False).reset_index(drop = True)
        
        ### Chart ###
        nbins = int(round(len(authorData['entropy'])/10,0))
        avg = authorData['entropy'].mean()
        std = stdev(authorData['entropy'])
        
        report = [
            'Mean entropy: {}.'.format(avg),
            'Standard deviation (entropy): {}.'.format(std),
            'Median size: {}'.format(authorData['size'].median())
        ]
        self.printTextReport('10a', report)
        
        plt.hist(authorData['entropy'], bins=nbins)
        plt.xlabel('Entropy')
        plt.ylabel('Number of clusters')
        
        plt.axvline(x=authorData['entropy'].mean(), color='r')
        plt.axhline(y=authorData['size'].median(), color='r')
        plt.xticks(list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, round(authorData['entropy'].mean(), 2), 0.7, 0.8, 0.9, 1.0]))
        plt.gca().get_xticklabels()[6].set_color('red')
        plt.yticks(list([0, 10, authorData['size'].median(), 20, 30, 40, 50, 60, 70, 80]))
        plt.gca().get_yticklabels()[2].set_color('red')

        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        
        self.savefig('10a')
        
    def observation10b(self, authorData):
        authorData = authorData.sort_values(by=['authors'], ascending=False).reset_index(drop = True)
        
        ### Chart ###
        nbins = int(round(len(authorData['authors'])/10,0))
        avg = authorData['authors'].mean()
        std = stdev(authorData['authors'])
        
        report = [
            'Mean (authors): {}.'.format(avg),
            'Standard deviation (authors): {}.'.format(std),
            'Median size: {}'.format(authorData['size'].median())
        ]
        self.printTextReport('10b', report)
        
        plt.hist(authorData['authors'], bins=nbins)
        plt.xlabel('Number of authors')
        plt.ylabel('Number of clusters')
        
        plt.axvline(x=authorData['authors'].mean(), color='r')
        plt.axhline(y=authorData['size'].median(), color='r')

        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        
        self.savefig('10b')
        
    def observation11(self):
        authorData = pd.read_pickle(f'{dataPath}/prepared/data_rq2.p')
        
        authorData = authorData[authorData['size']>=10]
        authorData = authorData.sort_values(by=['entropy'], ascending=False).reset_index(drop = True)
        
        authorData['sizeperc'] = authorData.apply(lambda row: round((row['size']/(authorData['size'].max())),2), axis=1)
        authorData = authorData.sort_values(by=['sizeperc'])
        
        ### Chart ###
        
        fig = plt.figure()
        ax = plt.gca()
        
        x = authorData['sizeperc']
        y = authorData['entropy']
        hb = ax.hexbin(x, y, gridsize = 35, cmap ='binary', edgecolor='gray', mincnt=1)
        
        plt.axvline(x=authorData['size'].median()/authorData['size'].max(), color='r')
        plt.axhline(y=authorData['entropy'].median(), color='r')
        plt.yticks(list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, round(authorData['entropy'].median(),2), 0.8, 0.9, 1.0]))
        plt.gca().get_yticklabels()[7].set_color('red')
        plt.xticks(list([0.00579, 0.2, 0.4, 0.6, 0.8, 1]))
        plt.gca().get_xticklabels()[0].set_color('red')
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        
        plt.xlabel('Normalized cluster size', fontsize=16)
        plt.ylabel('Entropy', fontsize=16)
        
        labels=ax.get_xticklabels()+ax.get_yticklabels()
        for label in labels:
            label.set_fontsize(16)
        
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Number of contracts', fontsize=16)
        cb.ax.tick_params(labelsize=16)
        
        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        
        self.savefig('11')
    
    def observation12(self):
        self.observation12a()
        plt.clf()
        self.observation12b()
    
    def observation12a(self):
        data = pd.read_csv(f'{dataPath}/clonedata/openzeppelin/type-1.csv')

        uniqueContractsWithIdenticalOZBlock = (data[data['startline_y'].notnull()].drop_duplicates(subset=['filename_x']))
        unique_contracts = data.drop_duplicates(subset=['filename_x'])

        report = [
            'Total number of contracts: {}.'.format(len(unique_contracts)),
            'Number of distinct contracts with an OpenZeppelin record associated: {}.'.format(len(uniqueContractsWithIdenticalOZBlock)),
            'Percentage ratio: {}%.'.format(round((len(uniqueContractsWithIdenticalOZBlock)/len(unique_contracts))*100, 2))
        ]
        self.printTextReport(12, report)

    def observation12b(self):
        data = pd.read_csv(f'{dataPath}/clonedata/openzeppelin/type-1_functions.csv')
        data['endline_x'] = data['endline_x'].apply( lambda x: int(x.replace('"', '')))
        data['startline_x'] = data['startline_x'].apply( lambda x: int(x.replace('"', '')))
        data['diff'] = data['endline_x'] - data['startline_x'] + 1

        report = []
            
        for i in range(11):
            data = data[data['diff'] >= i]

            uniqueContractsWithIdenticalOZBlock = (data[data['startline_y'].notnull()].drop_duplicates(subset=['filename_x']))
            unique_contracts = data.drop_duplicates(subset=['filename_x'])
            
            report.extend(
                (f'Min number of function lines considered: {i}',
                'Total number of functions: {}.'.format(len(unique_contracts)),
                'Number of distinct functions with an OpenZeppelin record associated: {}.'.format(len(uniqueContractsWithIdenticalOZBlock)),
                'Percentage ratio: {}%.'.format(round((len(uniqueContractsWithIdenticalOZBlock)/len(unique_contracts))*100, 2)),
                '\n')
            )
        
        self.printTextReport('12b', report)
    
    def observation13(self):
        self.observation13a()
        plt.clf()
        self.observation13b()
    
    def observation13a(self):
        data = pd.read_csv(f'{dataPath}/clonedata/openzeppelin/type-1.csv')
        all_corpus_contracts = pd.read_csv(f'{dataPath}/clonedata/openzeppelin/corpus_contracts.csv')

        openZeppelinRecords = (data[data['filename_y'].notnull()]).drop_duplicates(subset=['filename_x', 'startline_x', 'endline_x'])
        
        report = [
            'Total number of code blocks: {}.'.format(len(all_corpus_contracts)),
            'Number of distinct OpenZeppelin records: {}.'.format(len(openZeppelinRecords)),
            'Percentage ratio: {}%.'.format(round((len(openZeppelinRecords)/len(all_corpus_contracts))*100, 2))
        ]
        self.printTextReport(13, report)

    def observation13b(self):
        data = pd.read_csv(f'{dataPath}/clonedata/openzeppelin/type-1_functions.csv')
        data['endline_x'] = data['endline_x'].apply( lambda x: int(x.replace('"', '')))
        data['startline_x'] = data['startline_x'].apply( lambda x: int(x.replace('"', '')))
        data['diff'] = data['endline_x'] - data['startline_x'] + 1

        all_corpus_functions = pd.read_csv(f'{dataPath}/clonedata/openzeppelin/corpus_functions.csv')
        all_corpus_functions['endline'] = all_corpus_functions['endline'].apply( lambda x: int(x.replace('"', '')))
        all_corpus_functions['startline'] = all_corpus_functions['startline'].apply( lambda x: int(x.replace('"', '')))
        all_corpus_functions['diff'] = all_corpus_functions['endline'] - all_corpus_functions['startline'] + 1

        report = []
        for i in range(11):
            data = data[data['diff'] >= i]

            all_corpus_functions = all_corpus_functions[all_corpus_functions['diff'] >= i]
            openZeppelinRecords = (data[data['filename_y'].notnull()]).drop_duplicates(subset=['filename_x', 'startline_x', 'endline_x'])

            report.extend(
                (f'Min number of lines considered: {i}',
                'Total number of code blocks: {}.'.format(len(all_corpus_functions)),
                'Number of distinct OpenZeppelin records: {}.'.format(len(openZeppelinRecords)),
                'Percentage ratio: {}%.'.format(round((len(openZeppelinRecords)/len(all_corpus_functions))*100, 2)),
                '\n')
            )
        
        self.printTextReport('13b', report)

    def observation14(self):
        self.observation14a()
        plt.clf()
        self.observation14b()

    def observation14a(self):
        data = pd.read_csv(f'{dataPath}/clonedata/openzeppelin/type-1.csv')

        ozFiles = data.filename_y.apply(lambda x: x.replace('"','').split('/')[-1] if not pd.isna(x) else x)
        
        totalOZFiles = ozFiles.value_counts().sum()
        uniqueOZFileNames = len(ozFiles.unique())
        
        ozCounts = ozFiles.value_counts().rename_axis('contract').reset_index(name='count')
        
        ozCounts['cumulativeSum'] = ozCounts['count'].cumsum()
        ozCounts['perc'] = round((ozCounts['count']/totalOZFiles)*100, 2)
        ozCounts['cumulativePerc'] = round((ozCounts['cumulativeSum']/totalOZFiles)*100, 2)
        
        report = [
            ('Head 10', ozCounts.head(10), ''),
            ('Cumulative 80%', ozCounts.head(ozCounts[ozCounts.cumulativePerc > 80].index[0]), ''),
            ('First 20', ozCounts.head(int(len(ozCounts)*0.2)), '')
        ]        
        self.printHtmlReport(14, report)

    def observation14b(self):
        data = pd.read_csv(f'{dataPath}/clonedata/openzeppelin/type-1_functions.csv')

        data['endline_x'] = data['endline_x'].apply( lambda x: int(x.replace('"', '')))
        data['startline_x'] = data['startline_x'].apply( lambda x: int(x.replace('"', '')))
        data['diff'] = data['endline_x'] - data['startline_x'] + 1

        reports = []

        from collections import namedtuple
        report_with_cutoff = namedtuple('REPORT_WITH_CUTOFF', ('min_lines', 'report'))

        for i in range(11):
            data = data[data['diff'] >= i]

            ozFiles = data.function_id_y
            
            totalOZFiles = ozFiles.value_counts().sum()
            uniqueOZFileNames = len(ozFiles.unique())
            
            ozCounts = ozFiles.value_counts().rename_axis('functions').reset_index(name='count')
            
            ozCounts['cumulativeSum'] = ozCounts['count'].cumsum()
            ozCounts['perc'] = round((ozCounts['count']/totalOZFiles)*100, 2)
            ozCounts['cumulativePerc'] = round((ozCounts['cumulativeSum']/totalOZFiles)*100, 2)

            report = [
                ('Head 10', ozCounts.head(10), ''),
                ('Cumulative 80%', ozCounts.head(ozCounts[ozCounts.cumulativePerc > 80].index[0]), ''),
                ('First 20', ozCounts.head(int(len(ozCounts)*0.2)), '')
            ]

            reports.append(report_with_cutoff(i, report))
        
        self.printHtmlReport('14b', reports, functions=True)
    
    ########################## HELPER METHODS ##########################            
    def savefig(self, observationNumber):
        plt.gcf().tight_layout()
        plt.savefig('{}/observation{}.pdf'.format(resultsPath, observationNumber))
            
    def printTextReport(self, observationNumber, reports:list):
        f = open('{}/observation{}.txt'.format(resultsPath, observationNumber), 'w')
        for r in reports:
            f.write(r+'\n')
        f.close()
        
    def printHtmlReport(self, observationNumber, reports:list, functions=False):
        f = open('{}/observation{}.html'.format(resultsPath, observationNumber), 'w', encoding='utf-8')
        if functions:
            for cutoff_report in reports:
                cutoff = cutoff_report.min_lines
                report = cutoff_report.report
                f.write(f'<h1> Min_lines: {cutoff}</h1>')
                f.write('<div>')
                for title, df, comment in report:
                    f.write('<div style="margin-right:20px">')
                    f.write('<h2>{}</h2>'.format(title))
                    f.write(df.to_html())
                    f.write('<p>{}</p>'.format(comment))
                    f.write('</div>')
                f.write('</div>')
        else:
            f.write('<div>')
            for title, df, comment in reports:
                f.write('<div style="margin-right:20px">')
                f.write('<h2>{}</h2>'.format(title))
                f.write(df.to_html())
                f.write('<p>{}</p>'.format(comment))
                f.write('</div>')
            f.write('</div>')
        f.close()

    def runAll(self):
        print('Running analysis for all observations.\n')
        for observationId in range(1, 15):
            plt.clf()
            self.runOne(observationId)
    
    def runOne(self, observationId):
        print('Running analysis for observation{}.\n'.format(observationId))
        getattr(self, 'observation{}'.format(observationId))()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o','--observation', help='Observation ID.', type=str, nargs=1)
    parser.add_argument('-s','--stash', help='Stash results folder.', action='store_true')
    args = parser.parse_args()
    
    if args.stash:
        if os.path.exists(resultsPath) and os.path.isdir(resultsPath):
            shutil.rmtree(resultsPath)
        os.mkdir(resultsPath)
    
    analysis = Analysis()
    if not args.observation:
        analysis.runAll()
    else:
        analysis.runOne(args.observation[0])