#!/usr/bin/python

"""
util.py
Author: Daniel J Wilson, daniel.j.wilson@gmail.com

Utility functions for aDDM.
"""

import numpy as np
import pandas as pd
import sys

import plotly as plotly
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *

from plotly import tools
plotly.offline.init_notebook_mode(connected=True)


def plotly_combined(subject_num, subject_data, test_expdata, subj_params_df, sim_type=None):
    # Allows for offline plotting (importing up top now)
    # from plotly import tools
    # plotly.offline.init_notebook_mode(connected=True)

    subject = extract_subj_data(subject_num, subject_data)
    
    v = str(round(np.mean(test_expdata.est_scaling[test_expdata.subject == int(subject_num)]), 3))
    a = str(round(np.mean(test_expdata.est_boundary[test_expdata.subject == int(subject_num)]), 3))
    t = str(round(np.mean(test_expdata.est_theta[test_expdata.subject == int(subject_num)]), 3))

    if sim_type == "synth":
        mle = round(subj_params_df.loc[0, 'MLE'],2)
    else:
        mle = round(subj_params_df.loc[int(subject_num), 'MLE'],2)

    
    x = subject.quantile_means 

    # RT Means
    y1 = subject.test_rt_means       # test
    y2 = subject.sim_rt_means        # sim
    
    # Choice Means
    y3 = subject.test_choice_means   # test
    y4 = subject.sim_choice_means    # sim

    #------------------#
    # RT plot          #
    #------------------#
    
    # Test Data
    trace1 = Scatter(
        x=x,
        y=y1,
        error_y=dict(
                type='data',
                array=subject.test_rt_sems,
                color='rgb(0,170,80)',
                visible=True
        ),
        line=Line(color='rgb(0,100,80)'),
        mode='lines+markers',
        name='Test Data',
    )

    # Sim Data
    trace2 = Scatter(
        x=x,
        y=y2,
        error_y=dict(
                type='data',
                array=subject.sim_rt_sems,
                color='rgb(0,100,246)',
                visible=True
        ),
        line=Line(color='rgb(0,176,246)'),
        mode='lines+markers',
        name='Sim Data',
    )
    
    
    #------------------#
    # Choice plot      #
    #------------------#
    
    # Test Data
    trace3 = Scatter(
        x=x,
        y=y3,
        error_y=dict(
                type='data',
                array=subject.test_choice_sems,
                color='rgb(0,170,80)',
                visible=True
        ),
        line=Line(color='rgb(0,100,80)'),
        mode='lines+markers',
        name='Test Data',
        xaxis='x2',
        yaxis='y2',
    )

    # Sim Data
    trace4 = Scatter(
        x=x,
        y=y4,
        error_y=dict(
                type='data',
                array=subject.sim_choice_sems,
                color='rgb(0,100,246)',
                visible=True
        ),
        line=Line(color='rgb(0,176,246)'),
        mode='lines+markers',
        name='Sim Data',
        xaxis='x2',
        yaxis='y2',
    )
    

    data = Data([trace1, trace2, trace3, trace4])

    layout = Layout(
        paper_bgcolor='rgb(255,255,255)',
        plot_bgcolor='rgb(229,229,229)',
        
        xaxis=XAxis(
            gridcolor='rgb(255,255,255)',
            range=[-2,2],
            domain = [0,0.45],
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            title='Net Value ($)',
            zeroline=False
        ),
        xaxis2=XAxis(
            gridcolor='rgb(255,255,255)',
            range=[-2,2],
            domain = [0.55,1],
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            title='Net Value ($)',
            zeroline=False
        ),
        yaxis1=YAxis(
            gridcolor='rgb(255,255,255)',
            range=[0,],
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            title='RT (s)',
            anchor = 'x1',
            zeroline=False
        ),
        
        yaxis2=YAxis(
            gridcolor='rgb(255,255,255)',
            range=[0,],
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            title='p (accept)',
            anchor = 'x2',
            zeroline=False
        ),
        
        annotations=Annotations([
            Annotation(
            x=0.01,
            y=1.12,
            showarrow=False,
            text='PARAMETERS: v: ' + str(v) + ', a: ' + str(a) + ', θ: ' + str(t),
            xref='paper',
            yref='paper'
            ),
            Annotation(
            x=0.01,
            y=1.08,
            showarrow=False,
            text='MLE: ' + str(mle),
            xref='paper',
            yref='paper'
            ),
            Annotation(
            x=0.01,
            y=1.2,
            showarrow=False,
            text='SUBJECT: ' + str(subject_num),
            font=dict(
                family='Arial, sans-serif',
                size=22,
            ),
            xref='paper',
            yref='paper'
            )
        ])
    )

    # updatemenus=list([
    #     dict(
    #         buttons=list([   
    #             dict(
    #                 label='00',
    #                 method='update',
    #                 args=[extract_subj_data(0, subject_data)]
    #             ),
    #             dict(
    #                 label='01',
    #                 method='update',
    #                 args=[extract_subj_data(1, subject_data)]
    #             )             
    #         ]),
    #         direction = 'down',
    #         pad = {'r': 10, 't': 10},
    #         showactive = True,
    #         x = 0.1,
    #         xanchor = 'left',
    #         y = 1.1,
    #         yanchor = 'top' 
    #     ),
    # ])

    # annotations = list([
    #     dict(text='Subject:' + str(subject_num), x=0, y=1.085, yref='paper', align='left', showarrow=False)
    # ])


    # layout['updatemenus'] = updatemenus
    # layout['annotations'] = annotations
    
#     fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('RT', 'Choice'))
    
#     fig.append_trace([trace1, trace2], 1, 1)
#     fig.append_trace([trace3, trace4], 1, 2)
    
#     fig['layout']['xaxis1'].update(title='xaxis 1 title')
#     fig['layout']['xaxis2'].update(title='xaxis 2 title')
    
#     fig['layout']['yaxis1'].update(title='yaxis 1 title')
#     fig['layout']['yaxis2'].update(title='yaxis 2 title')

    fig = Figure(data=data, layout=layout)
    plotly.offline.iplot(fig, filename= 'shaded_lines')   




def plotly_rt(subject_num, subject_data):

    plotly.offline.init_notebook_mode(connected=True)

    subject = extract_subj_data(subject_num, subject_data)

    x = subject.quantile_means 

    # Test Data: RT Means
    y1 = subject.test_rt_means         # test
    y2 = subject.sim_rt_means
    
    # Choice Means
    y3 = subject.test_choice_means   # test
    y4 = subject.sim_choice_means    # sim

    # Test Data
    trace1 = Scatter(
        x=x,
        y=y1,
        error_y=dict(
                type='data',
                array=subject.test_rt_sems,
                visible=True
        ),
        line=Line(color='rgb(0,100,80)'),
        mode='lines+markers',
        name='Test Data',
    )

    # Sim Data
    trace2 = Scatter(
        x=x,
        y=y2,
        error_y=dict(
                type='data',
                array=subject.sim_rt_sems,
                visible=True
        ),
        line=Line(color='rgb(0,176,246)'),
        mode='lines+markers',
        name='Sim Data',
    )


    data = Data([trace1, trace2])

    layout = Layout(
        paper_bgcolor='rgb(255,255,255)',
        plot_bgcolor='rgb(229,229,229)',
        xaxis=XAxis(
            gridcolor='rgb(255,255,255)',
            range=[-2,2],
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            title='Net Value ($)',
            zeroline=False
        ),
        yaxis=YAxis(
            gridcolor='rgb(255,255,255)',
            range=[0,],
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            title='RT (s)',
            zeroline=False
        ),
        title='Subject ' + str(subject_num),
    )

    # updatemenus=list([
    #     dict(
    #         buttons=list([   
    #             dict(
    #                 label='00',
    #                 method='update',
    #                 args=[extract_subj_data(0, subject_data)]
    #             ),
    #             dict(
    #                 label='01',
    #                 method='update',
    #                 args=[extract_subj_data(1, subject_data)]
    #             )             
    #         ]),
    #         direction = 'down',
    #         pad = {'r': 10, 't': 10},
    #         showactive = True,
    #         x = 0.1,
    #         xanchor = 'left',
    #         y = 1.1,
    #         yanchor = 'top' 
    #     ),
    # ])

    # annotations = list([
    #     dict(text='Subject:' + str(subject_num), x=0, y=1.085, yref='paper', align='left', showarrow=False)
    # ])


    # layout['updatemenus'] = updatemenus
    # layout['annotations'] = annotations

    fig = Figure(data=data, layout=layout)
    plotly.offline.iplot(fig, filename= 'shaded_lines')   


def plotly_choice(subject_num, subject_data):

    plotly.offline.init_notebook_mode(connected=True)

    subject = extract_subj_data(subject_num, subject_data)

    x = subject.quantile_means 
    x_rev = x[::-1]

    # Test Data: RT Means
    y1 = subject.test_choice_means         # mean RT
    y1_upper = subject.test_choice_means + subject.test_choice_sems  # add SEM
    y1_lower = subject.test_choice_means - subject.test_choice_sems  # subtract SEM
    y1_lower = y1_lower[::-1] # necessary??

    # Sim Data: RT Means
    y2 = subject.sim_choice_means
    y2_upper = subject.sim_choice_means + subject.sim_choice_sems
    y2_lower = subject.sim_choice_means - subject.sim_choice_sems
    y2_lower = y2_lower[::-1]


    # Test Data
    trace1 = Scatter(
        x=x,
        y=y1,
        error_y=dict(
                type='data',
                array=subject.test_choice_sems,
                visible=True
        ),
        line=Line(color='rgb(0,100,80)'),
        mode='lines+markers',
        name='Test Data',
    )

    # Sim Data
    trace2 = Scatter(
        x=x,
        y=y2,
        error_y=dict(
                type='data',
                array=subject.sim_choice_sems,
                visible=True
        ),
        line=Line(color='rgb(0,176,246)'),
        mode='lines+markers',
        name='Sim Data',
    )


    data = Data([trace1, trace2])

    layout = Layout(
        paper_bgcolor='rgb(255,255,255)',
        plot_bgcolor='rgb(229,229,229)',
        xaxis=XAxis(
            gridcolor='rgb(255,255,255)',
            range=[-2,2],
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            title='Net Value ($)',
            zeroline=False
        ),
        yaxis=YAxis(
            gridcolor='rgb(255,255,255)',
            range=[0,],
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            title='p(accept)',
            zeroline=False
        ),
        title='Subject ' + str(subject_num),
    )

    # updatemenus=list([
    #     dict(
    #         buttons=list([   
    #             dict(
    #                 label='00',
    #                 method='update',
    #                 args=[extract_subj_data(0, subject_data)]
    #             ),
    #             dict(
    #                 label='01',
    #                 method='update',
    #                 args=[extract_subj_data(1, subject_data)]
    #             )             
    #         ]),
    #         direction = 'down',
    #         pad = {'r': 10, 't': 10},
    #         showactive = True,
    #         x = 0.1,
    #         xanchor = 'left',
    #         y = 1.1,
    #         yanchor = 'top' 
    #     ),
    # ])

    # annotations = list([
    #     dict(text='Subject:' + str(subject_num), x=0, y=1.085, yref='paper', align='left', showarrow=False)
    # ])


    # layout['updatemenus'] = updatemenus
    # layout['annotations'] = annotations

    fig = Figure(data=data, layout=layout)
    plotly.offline.iplot(fig, filename= 'shaded_lines')   


# format data
def format_for_plotting(subjects, test_expdata, subj_params_df):
    """
    """
    
    test_data = test_expdata
    # Create the variables (summed value) that we are using for our quantiles
    x = subjects.summed_val
    x1 = test_data.summed_val

    # Define quantiles - bins are [a, b) 
    bins = np.array([-4, -1., -0.5, -0.15, 0.15, 0.5, 1., 4])

    # Create column which indicates quantile of row
    subjects['quantile'] = np.digitize(x, bins)
    test_data['quantile'] = np.digitize(x1, bins)
    # recode -1 response (reject) as 0 for probability of acceptance calculation
    subjects.resp[subjects.resp == -1.] = 0
    subjects.sim_resp[subjects.sim_resp == -1.] = 0
    test_data.resp[test_data.resp == -1.] = 0
    
    
    #--------------------------------#
    # descriptive stats by subject   #
    #--------------------------------#
    
    test_rt_stats = pd.DataFrame(test_data.groupby(['subject', 'quantile']).rt.agg(['mean', 'median', 'std', 'sem']))
    test_choice_stats = pd.DataFrame(test_data.groupby(['subject', 'quantile']).resp.agg(['mean', 'std', 'sem']))

    sim_rt_stats = pd.DataFrame(subjects.groupby(['subject', 'quantile']).sim_rt.agg(['mean', 'median', 'std', 'sem']))
    sim_choice_stats = pd.DataFrame(subjects.groupby(['subject', 'quantile']).sim_resp.agg(['mean', 'std', 'sem']))

    summed_val_sim = pd.DataFrame(subjects.groupby(['subject', 'quantile']).summed_val.agg(['mean']))
    summed_val_test = pd.DataFrame(test_data.groupby(['subject', 'quantile']).summed_val.agg(['mean']))

    test_rt_stats.rename(index=str, columns={"mean": "test_rt_mean", "median": "test_rt_median", "std": "test_rt_std", "sem": "test_rt_sem"}, inplace=True)
    test_choice_stats.rename(index=str, columns={"mean": "test_choice_mean", "median": "test_choice_median", "std": "test_choice_std", "sem": "test_choice_sem"}, inplace=True)

    sim_rt_stats.rename(index=str, columns={"mean": "sim_rt_mean", "median": "sim_rt_median", "std": "sim_rt_std", "sem": "sim_rt_sem"}, inplace=True)
    sim_choice_stats.rename(index=str, columns={"mean": "sim_choice_mean", "median": "sim_choice_median", "std": "sim_choice_std", "sem": "sim_choice_sem"}, inplace=True)

    summed_val_test.rename(index=str, columns={"mean": "test_summed_val_mean"}, inplace=True)
    summed_val_sim.rename(index=str, columns={"mean": "sim_summed_val_mean"}, inplace=True)
    
    
    #--------------------------------#
    # merge rt and choice stats      #
    #--------------------------------#
    
    test_quantized_subj_stats = pd.concat([test_rt_stats, test_choice_stats, summed_val_test], axis = 1)
    sim_quantized_subj_stats = pd.concat([sim_rt_stats, sim_choice_stats, summed_val_sim], axis = 1)
    test_sim_data = pd.concat([test_quantized_subj_stats, sim_quantized_subj_stats], axis = 1)
    
    return test_sim_data



def extract_subj_data(subj_num, data):
    """
    """
    
    subject = str(subj_num)

    quantile_means = data.loc[subject].loc[:,'test_summed_val_mean'] # maps to quantile means

    # RTs
    # Test
    test_rt_means = data.loc[subject].loc[:,'test_rt_mean']
    test_rt_sems = data.loc[subject].loc[:,'test_rt_sem']
    # Sim
    sim_rt_means = data.loc[subject].loc[:,'sim_rt_mean']
    sim_rt_sems = data.loc[subject].loc[:,'sim_rt_sem']

    # Choices
    # Test
    test_choice_means = data.loc[subject].loc[:,'test_choice_mean']
    test_choice_sems = data.loc[subject].loc[:,'test_choice_sem']
    # Sim
    sim_choice_means = data.loc[subject].loc[:,'sim_choice_mean']
    sim_choice_sems = data.loc[subject].loc[:,'sim_choice_sem']
    
    
    #--------------------------------------------#
    # Create DF with stats from single subject   #
    #--------------------------------------------#
    
    subject_stats = pd.DataFrame(quantile_means)

    # RT
    subject_stats['test_rt_means'] = test_rt_means
    subject_stats['test_rt_sems'] = test_rt_sems
    subject_stats['sim_rt_means'] = sim_rt_means
    subject_stats['sim_rt_sems'] = sim_rt_sems

    # Choice
    subject_stats['test_choice_means'] = test_choice_means
    subject_stats['test_choice_sems'] = test_choice_sems
    subject_stats['sim_choice_means'] = sim_choice_means
    subject_stats['sim_choice_sems'] = sim_choice_sems

    subject_stats = subject_stats.rename(columns = {
        'test_summed_val_mean':'quantile_means'
    })
    #subject_stats.reset_index(drop=True, inplace=True)
    return subject_stats