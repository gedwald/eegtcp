#!/usr/bin/env python
# -*- coding: utf-8 -*-
#============================================================================
# Code: NIBS_Paradigm
# Author: Jiachen XU <jiachen.xu.94@gmail.com>
#
# Last Update: 2020-06-28
#============================================================================

from __future__ import absolute_import, division
import pdb
from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard
import pdb
# What signaler class to use? Here just the demo signaler:
from psychopy.voicekey.demo_vks import DemoVoiceKeySignal as Signaler
import sounddevice as sd
from scipy.io.wavfile import write

from nibs_func import *
from jxu.hardware.signal import SignalGenerator as SG
import time
from datetime import datetime
import logging as jxu_logging
import random

# -----------------------------------------------------------------------------------
# ------------------------- Setting: Parameter --------------------------------------
# -----------------------------------------------------------------------------------

######## Psychopy basic setting ########
endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame
continue_str = '          Press [space] key to continue.'
font_type = 'Airal'

# will be replaced as event_dict from jxu.data.utils.nibs_event_dict()[2]
event_dict = {'ESC': 1,
              'Test': 253,
              'Main': 254,
              'End':255,
              'Pre_run': [2, 3],
              'Post_run': [8, 9],
              'Run': [4, 5],
              'Block': [6, 7],
              'Cali_intro': [10, 11],
              'Cali_trial': [12, 13],
              'Cali_display': [14, 15],
              'Cali_ans': [16, 17],
              'Cali_rec': [18, 19],
              'Stim': [20, 21],
              'Sham': [22, 23],
              'Fade_in': [24, 25],
              'Fade_out': [26, 27],
              'Stable_stim': [28, 29],
              'RS_intro': [30, 31],
              'RS_open': [32, 33],
              'RS_close': [34, 35],
              'QA_intro': [40, 41],
              'QA_trial': [42, 43],
              'QA_audio': [44, 45],
              'QA_ans': [46, 47],
              'QA_rec': [48, 49],
              'QA_cen_word': [50, 51],
              'Pause': [60, 61],
              'Break': [62, 63],
              'Arti_intro': [70, 71],
              'Arti_trial': [72, 73],
              'Arti_action': [74, 75],
              'Arti_rec': [78, 79]}

instruction_pos = [0.5, 0.1]
instruction_annot_pos = [0.5, -0.4]
title_pos = [0.5, 0.4]
qa_inrto_title_pos = [0.5, 0.5]
text_pos = [0.5, 0]
annot_pos = [0.5, 0.0]
audio_root = '/home/jxu/File/Data/NIBS/Stage_one/Audio/Soundeffect/'
question_root = '/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/'

######## Paradigm setting ########
n_run = 4
n_block = 2
n_trial = 25  # 25 - 500 secs
n_cali_trial = 10  # 10 - 200 secs
n_arti_trial = 15 # 15 - 300 secs

n_question = n_run * n_block * n_trial + n_cali_trial * 2

fs = 44100  # Sample rate of audio !!! For recording not playing!!!
n_rec_chn = 1
 
stim_run = [1, 2]  # In which run, the stimulation is applied.
init_intensity, min_intensity, max_intensity = 0.002, 0.005, 0.5
intensity_goal = [0, max_intensity, max_intensity, 0]  # i.e., of each run
stim_freq = 0.0
fade_in_auto_incre = 6 
n_step_fade_stim = 5

step_intensity = max_intensity / n_step_fade_stim
    
input_intensity = init_intensity
output_intensity = init_intensity

######## Component switch ########
init_flag = True  # NEVER TURN OFF THIS FLAG!!! For initializing the components

external_question_flag = 1
cali_question_cnt = -1
qa_question_cnt = -1
artifact_action_cnt = -1


word_type = 'VERB'

fade_in_out_show = False   
always_dislpay = True
flexible_qa_rec_start = True

comp_gap = 0.4
instruction_cont_start, instruction_cont_dur = 15, None  # Audio 14s


artifact_intro_start, artifact_intro_dur, artifact_intro_cont_dur = 0, 27, None  # Audio 26s
artifact_intro_cont_start = artifact_intro_start + artifact_intro_dur + comp_gap

artifact_total_time = 20.0
artifact_hint_start, artifact_hint_dur, artifact_action_dur, artifact_a_beep_s_dur, artifact_rec_dur, artifact_a_beep_e_dur, artifact_break_dur = 0, 2, 3, 1, 10, 1, 1
artifact_action_start = artifact_hint_start + artifact_hint_dur + comp_gap
artifact_a_beep_s_start = artifact_action_start + artifact_action_dur + comp_gap
artifact_rec_start = artifact_a_beep_s_start + artifact_a_beep_s_dur + comp_gap
artifact_a_beep_e_start = artifact_rec_start + artifact_rec_dur + comp_gap
artifact_break_start = artifact_a_beep_e_start + artifact_a_beep_e_dur + comp_gap
artifact_text_dur = artifact_break_start + artifact_break_dur


cali_intro_start, cali_intro_dur, cali_intro_cont_dur = 0, 30, None  # Audio 29s
cali_intro_cont_start = cali_intro_start + cali_intro_dur  + comp_gap

# 30s version: 0, 2, 5, 1, 17, 1, 2
# 20s version: 0, 2, 1, 1, 11, 1, 2
cali_total_time = 20.0
cali_hint_start, cali_hint_dur, cali_q_dur, cali_a_beep_s_dur, cali_rec_dur, cali_a_beep_e_dur, cali_break_dur = 0, 2, 1, 1, 11, 1, 2
cali_q_start = cali_hint_start + cali_hint_dur + comp_gap
cali_a_beep_s_start = cali_q_start + cali_q_dur + comp_gap
cali_rec_start = cali_a_beep_s_start + cali_a_beep_s_dur + comp_gap
cali_a_beep_e_start = cali_rec_start + cali_rec_dur + comp_gap
cali_break_start = cali_a_beep_e_start + cali_a_beep_e_dur + comp_gap
cali_text_dur = cali_break_start + cali_break_dur

# Previous cali_q_dur is for sequencially setting subsequent comp, this is the true diaplay
if always_dislpay:
    cali_q_dur = cali_text_dur - cali_q_start



rs_intro_text_start, rs_intro_text_dur, rs_intro_cont_dur = 0, 21, None    # Audio 20s
rs_intro_cont_start = rs_intro_text_start + rs_intro_text_dur + comp_gap

rs_rec_text_start, rs_rec_text_dur, rs_rec_beep_e_dur, rs_rec_cont_dur = 0, 180, 1, None     # 180s!
rs_rec_beep_e_start = rs_rec_text_start + rs_rec_text_dur + comp_gap
rs_rec_cont_start = rs_rec_beep_e_start + rs_rec_beep_e_dur + comp_gap

QA_intro_title_start, QA_intro_title_dur, QA_intro_cont_dur = 0, 40, None # Audio 39s
QA_intro_cont_start = QA_intro_title_start + QA_intro_title_dur + comp_gap


# 30s version: 0, 2, 14, 1, 8, 1, 2
# 25s version: 0, 2, 12, 1, 6, 1, 1
# 20s version: 0, 2, 5, 1, 8, 1, 1
qa_total_time = 20.0
QA_hint_start, QA_hint_dur, QA_q_dur, QA_a_beep_s_dur, QA_rec_dur, QA_a_beep_e_dur, QA_break_dur = 0, 2, 5, 1, 8, 1, 1
QA_q_start = QA_hint_start + QA_hint_dur + comp_gap
QA_a_beep_s_start = QA_q_start + QA_q_dur + comp_gap
QA_rec_start = QA_a_beep_s_start + QA_a_beep_s_dur + comp_gap
QA_a_beep_e_start = QA_rec_start + QA_rec_dur + comp_gap
QA_break_start = QA_a_beep_e_start + QA_a_beep_e_dur + comp_gap
QA_text_dur = QA_break_start + QA_break_dur

if flexible_qa_rec_start and external_question_flag:
    QA_trial_dur = qa_total_time
    QA_text_dur = QA_trial_dur
    QA_break_start = QA_text_dur - QA_break_dur
    QA_a_beep_e_start = QA_break_start - QA_a_beep_e_dur - comp_gap
    # beep_s_start is defined differently depending on sentence length.


trigger_sending(event_dict['Main'], default_sleep=True)  # trigger 254 represents main experiment
# -----------------------------------------------------------------------------------
# ------------------------ Setting: Initialization ----------------------------------
# -----------------------------------------------------------------------------------
if init_flag:

    expInfo, win, frameDur, defaultKeyboard  = exp_init()
    folder_path, filename = path_init(expInfo)

    font_size = (0.06 if not expInfo['Full_screen'] else 0.15)


    Instruction_flag = expInfo['Instruction']
    Intro_flag = expInfo['Intro']
    Artifact_intro_flag = expInfo['Artifact']  & Intro_flag
    Artifact_rec_flag = expInfo['Artifact']
    Artifact_intro_within_flag = expInfo['Artifact_within']  & Intro_flag
    Artifact_rec_within_flag = expInfo['Artifact_within']
    Cali_de_pre_intro_flag = expInfo['Cali_pre']  & Intro_flag
    Cali_de_pre_rec_flag = expInfo['Cali_pre']
    fade_in_flag = expInfo['Fade_in_out']
    fade_out_flag = expInfo['Fade_in_out']
    RS_intro_flag = expInfo['Resting_State'] & Intro_flag
    RS_rec_flag = expInfo['Resting_State']
    QA_intro_flag = expInfo['QA']  & Intro_flag
    QA_rec_flag = expInfo['QA']
    Pause_flag = expInfo['Pause']
    Cali_de_post_intro_flag = expInfo['Cali_post'] & Intro_flag
    Cali_de_post_rec_flag = expInfo['Cali_post']
    end_flag = expInfo['End']

    break_cali_pre_trial = (None if expInfo['Breakpoint_Cali_pre_trial'] == 'No' else int(expInfo['Breakpoint_Cali_pre_trial']))
    break_cali_post_trial = (None if expInfo['Breakpoint_Cali_post_trial'] == 'No' else int(expInfo['Breakpoint_Cali_post_trial']))
    break_run = (None if expInfo['Breakpoint_Run'] == 'No' else int(expInfo['Breakpoint_Run']))
    break_rs_block = (None if expInfo['Breakpoint_RS_block'] == 'No' else int(expInfo['Breakpoint_RS_block']))
    break_qa_block = (None if expInfo['Breakpoint_QA_block'] == 'No' else int(expInfo['Breakpoint_QA_block']))
    break_qa_trial = (None if expInfo['Breakpoint_QA_trial'] == 'No' else int(expInfo['Breakpoint_QA_trial']))

    if external_question_flag:

        pre_load_qa_df = pd.read_pickle(
            './qa_info/S{0}_Session{1}_unshattered_beep_df.pkl'.format(
                str(int(expInfo['participant'])).zfill(2), str(int(expInfo['session'])) ))
        extract_df, question_path, censor_question_start, censor_question_duration, sen_duration, sen_text, cen_text = extract_qa(
            input_all_df=pre_load_qa_df)

        # extract_df, question_path, censor_question_start, censor_question_duration, sen_duration, sen_text, cen_text
        pre_load_cali_df = pd.read_pickle(
            './cali_info/S{0}_Session{1}_cali.pkl'.format(str(int(expInfo['participant'])).zfill(2),
                                                          str(int(expInfo['session']))))
        cali_sen_text, cali_sen_order = extract_cali(input_all_df=pre_load_cali_df)

        arti_file_name_list = ['eye_horizontal.wav',
             'eye_vertical.wav',
             'eye_circular.wav',
             'eye_blink.wav',
             'frown.wav',
             'head_horizontal.wav',
             'head_vertical.wav',
             'arm.wav',
             'leg.wav',
             'fist.wav',
             'wrist.wav',
             'upper_body.wav',
             'teeth.wav',
             'no_respiration.wav',
             'respiration.wav']
        action_path_list = [audio_root + 'artifact/' + arti_file for arti_file in arti_file_name_list]



    # save a log file for key info
    now = datetime.now()
    time_exp_start = now.strftime("%Y_%m_%d_%H_%M%S")
    log_name = '/home/jxu/File/Experiment/NIBS/Sync/NIBS_paradigm/minimal_expfile/jxu_log/' + \
        'Subject_' + expInfo['participant'] +'_Session_' + expInfo['session'] + '_' + time_exp_start + '.log'
    jxu_logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s  %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S +0000',
                            filename=log_name,
                            filemode='a')

    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

    instruction_text_str = '     Thank you for participating in the experiment: \n\n' + \
        '     Personalized non-invasive brain stimulation    \n' + \
        '            for second language learning            \n'

    instruction_comp_list = [
        textstim_generator(win=win, name='text', content=instruction_text_str, pos=instruction_pos),
        audio_generator(name='audio', loc=audio_root+'Instruction_main.wav', secs=-1),
        key_resp_generator(name='key_resp'),
        textstim_generator(win=win, name='cont', content=continue_str, pos=instruction_annot_pos)
        ]
    instruction = routine_init('instruction', instruction_comp_list)
    instruction['time'] = {'text':[0, instruction_cont_dur],
                           'audio':[0, instruction_cont_dur],
                           'key_resp':[instruction_cont_start, instruction_cont_dur],
                           'cont':[instruction_cont_start, instruction_cont_dur]}


    # Initialize components for Routine "Artifact_intro"
    Artifact_intro_text_str = '  - Task: Artifacts collection   \n' + \
        '  - Please follow audio instruction              \n'

    Artifact_intro_comp_list = [
        textstim_generator(win=win, name='title', content='Muscular signal recording', pos=title_pos),
        textstim_generator(win=win, name='text', content=Artifact_intro_text_str, pos=text_pos),
        audio_generator(name='audio', loc=audio_root+'artifact/intro.wav', secs=-1),
        key_resp_generator(name='key_resp'),
        textstim_generator(win=win, name='cont', content=continue_str, pos=annot_pos)
        ]
    Artifact_intro = routine_init('Artifact_intro', Artifact_intro_comp_list)
    Artifact_intro['time'] = {'title':[artifact_intro_start, artifact_intro_dur],
                              'text':[artifact_intro_start, artifact_intro_dur],
                              'audio':[artifact_intro_start, artifact_intro_dur],
                              'key_resp':[artifact_intro_cont_start, artifact_intro_cont_dur],
                              'cont':[artifact_intro_cont_start, artifact_intro_cont_dur]}

    # Initialize components for Routine "Artifact_rec"
    Artifact_rec_text_str = ''  # Please read out following sentence.
    Artifact_rec_comp_list = [
        textstim_generator(win=win, name='text', content=Artifact_rec_text_str, pos=title_pos),
        audio_generator(name='beep_hint', loc=audio_root+'artifact/reminder.wav', secs=0.6),
        audio_generator(name='action', loc=audio_root+'artifact/reminder.wav', secs=0.6),
        audio_generator(name='beep_start', loc=audio_root+'artifact/C3A_C4A_tone_decrease_1s_new.wav', secs=1),
        textstim_generator(win=win, name='recording', content='', pos=instruction_annot_pos),
        audio_generator(name='beep_end', loc=audio_root+'artifact/C4A_C3A_tone_decrease_1s_new.wav', secs=1),
        textstim_generator(win=win, name='break', content='Short break', pos=instruction_annot_pos)
        ]

    Artifact_rec = routine_init('Artifact_rec', Artifact_rec_comp_list)
    Artifact_rec['time'] = {'text':[0, artifact_text_dur],
                            'beep_hint':[artifact_hint_start, artifact_hint_dur],
                            'action':[artifact_action_start, artifact_action_dur],
                            'beep_start':[artifact_a_beep_s_start, artifact_a_beep_s_dur],
                            'recording':[artifact_rec_start, artifact_rec_dur],
                            'beep_end':[artifact_a_beep_e_start, artifact_a_beep_e_dur],
                            'break':[artifact_break_start, artifact_break_dur]}

    # Initialize components for Routine "Cali_de_pre_intro
    Cali_de_pre_intro_text_str = '  - Task: Read out the displayed German sentence   \n' + \
        '  - Melodious beep sound: Trial start              \n' + \
        '  - Beep sound with increasing pitch: Rec. start   \n' + \
        '  - Beep sound with decreasing pitch: Rec. finish  \n'

    Cali_de_pre_intro_comp_list = [
        textstim_generator(win=win, name='title', content='READ OUT BLOCK (GERMAN)', pos=title_pos),
        textstim_generator(win=win, name='text', content=Cali_de_pre_intro_text_str, pos=text_pos),
        audio_generator(name='audio', loc=audio_root+'calibration/calibration.wav', secs=-1),
        key_resp_generator(name='key_resp'),
        textstim_generator(win=win, name='cont', content=continue_str, pos=annot_pos)
        ]
    Cali_de_pre_intro = routine_init('Cali_de_pre_intro', Cali_de_pre_intro_comp_list)
    Cali_de_pre_intro['time'] = {'title':[cali_intro_start, cali_intro_dur],
                                 'text':[cali_intro_start, cali_intro_dur],
                                 'audio':[cali_intro_start, cali_intro_dur],
                                 'key_resp':[cali_intro_cont_start, cali_intro_cont_dur],
                                 'cont':[cali_intro_cont_start, cali_intro_cont_dur]}

    # Initialize components for Routine "Cali_de_pre_rec"
    Cali_de_pre_rec_text_str = ''  # Please read out following sentence.
    Cali_de_pre_rec_comp_list = [
        textstim_generator(win=win, name='text', content=Cali_de_pre_rec_text_str, pos=title_pos),
        audio_generator(name='beep_hint', loc=audio_root+'calibration/reminder.wav', secs=0.6),
        textstim_generator(win=win, name='question_text', content=Cali_de_pre_rec_text_str, pos=text_pos),
        audio_generator(name='beep_start', loc=audio_root+'calibration/C3A_C4A_tone_decrease_1s_new.wav', secs=1),
        rec_generator(name='recording', sps=fs, loc='./data/', n_rec_chn=n_rec_chn),
        audio_generator(name='beep_end', loc=audio_root+'calibration/C4A_C3A_tone_decrease_1s_new.wav', secs=1),
        textstim_generator(win=win, name='break', content='Short break', pos=instruction_annot_pos)
        ]

    Cali_de_pre_rec = routine_init('Cali_de_pre_rec', Cali_de_pre_rec_comp_list)
    Cali_de_pre_rec['time'] = {'text':[0, cali_text_dur],
                               'beep_hint':[cali_hint_start, cali_hint_dur],
                               'question_text':[cali_q_start, cali_q_dur],
                               'beep_start':[cali_a_beep_s_start, cali_a_beep_s_dur],
                               'recording':[cali_rec_start, cali_rec_dur],
                               'beep_end':[cali_a_beep_e_start, cali_a_beep_e_dur],
                               'break':[cali_break_start, cali_break_dur]}

    # Initialize components for Routine "fade_in"
    fade_str_func = lambda x: 'The stimulation is starting now and we will gradually increase the intensity to '  + \
        str(x*2) + 'mA. \n\n Current current intensity is '

    fade_in_str = '' #  fade_str_func(max_intensity)
    fade_cont_str = 'Increase current intensity press i. \n' + \
        'Remain current intensity and start stimulation press space key. \n' + \
        'Decrease current intensity press d \n'
    fade_in_list = [
        textstim_generator(win=win, name='text', content=fade_in_str, pos=[0.5, 0.4]),
        key_resp_generator(name='auto_stim')
        ]
    fade_in = routine_init('fade_in', fade_in_list)
    fade_in['time'] = {'text':[0, fade_in_auto_incre], 'auto_stim':[0, fade_in_auto_incre]}


    # Initialize components for Routine "RS_intro"
    RS_intro_text_str = ' - Task: Keep seated and relaxed with either      \n' + \
        '             opening or closing eyes                  \n' + \
        ' - Beep sound with decreasing pitch: Block finish \n'

    RS_intro_comp_list = [
        textstim_generator(win=win, name='title', content='REST STATE BLOCK', pos=title_pos),
        textstim_generator(win=win, name='text', content=RS_intro_text_str, pos=text_pos),
        audio_generator(name='audio', loc=audio_root+'resting_state/rs.wav', secs=-1)
        ]
    RS_intro = routine_init('RS_intro', RS_intro_comp_list)
    RS_intro['time'] = {'title':[rs_intro_text_start, rs_intro_text_dur],
                        'text':[rs_intro_text_start, rs_intro_text_dur],
                        'audio':[rs_intro_text_start, rs_intro_text_dur]}

    # Initialize components for Routine "RS_rec"
    RS_rec_text_str = 'Please keep relaxed and open your eyes.\nNote: Blinking is allowed.'
    RS_rec_comp_list = [
        textstim_generator(win=win, name='text', content=RS_rec_text_str, pos=text_pos),
        audio_generator(name='beep_end', loc=audio_root+'resting_state/C4A_C3A_tone_decrease_1s_new.wav', secs=1)
        ]
    RS_rec = routine_init('RS_rec', RS_rec_comp_list)
    RS_rec['time'] = {'text':[rs_rec_text_start, rs_rec_text_dur],
                      'beep_end': [rs_rec_beep_e_start, rs_rec_beep_e_dur]}

    # Initialize components for Routine "QA_intro"
    QA_intro_text_str = ' - Task: Listen to the question& Speak out answer \n' + \
        ' - Melodious beep sound: Trial start              \n' + \
        ' - Silent sound (blank): censored word            \n' + \
        ' - Beep sound with increasing pitch: Rec. start   \n' + \
        ' - Beep sound with decreasing pitch: Rec. finish  \n'
    QA_intro_comp_list = [
        textstim_generator(win=win, name='title', content='Q&A BLOCK', pos=qa_inrto_title_pos),
        textstim_generator(win=win, name='text', content=QA_intro_text_str, pos=text_pos),
        audio_generator(name='audio', loc=audio_root+'q_a_40Hz/q_a_update_assr.wav', secs=-1),
        key_resp_generator(name='key_resp'),
        textstim_generator(win=win, name='cont', content=continue_str, pos=annot_pos)
        ]
    QA_intro = routine_init('QA_intro', QA_intro_comp_list)
    QA_intro['time'] = {'title':[QA_intro_title_start, QA_intro_title_dur],
                        'text':[QA_intro_title_start, QA_intro_title_dur],
                        'audio':[QA_intro_title_start, QA_intro_title_dur],
                        'key_resp':[QA_intro_cont_start, QA_intro_cont_dur],
                        'cont':[QA_intro_cont_start, QA_intro_cont_dur]}


    # Initialize components for Routine "QA_rec"
    QA_rec_text_str = 'Listen to the question and speak out your answer!'
    QA_rec_comp_list = [
        textstim_generator(win=win, name='text', content=QA_rec_text_str, pos=text_pos),
        audio_generator(name='beep_hint', loc=audio_root+'q_a_40Hz/reminder.wav', secs=0.6),
        audio_generator(name='question', loc=question_root+'old_data/article_0/sentence_0/sentence_0_syn_44100.wav', secs=1),
        audio_generator(name='beep_start', loc=audio_root+'q_a_40Hz/C3A_C4A_tone_decrease_1s_new.wav', secs=1),
        rec_generator(name='recording', sps=fs, loc='./data/', n_rec_chn=n_rec_chn),
        audio_generator(name='beep_end', loc=audio_root+'q_a_40Hz/C4A_C3A_tone_decrease_1s_new.wav', secs=1),
        textstim_generator(win=win, name='break', content='Short break', pos=instruction_annot_pos),
        trigger_generator(win=win, name='censor_word')
        ]
    QA_rec = routine_init('QA_rec', QA_rec_comp_list)
    QA_rec['time'] = {'text':[0, QA_text_dur],
                      'beep_hint':[QA_hint_start, QA_hint_dur],
                      'question':[QA_q_start, QA_q_dur],
                      'beep_start':[QA_a_beep_s_start, QA_a_beep_s_dur],
                      'recording':[QA_rec_start, QA_rec_dur],
                      'beep_end':[QA_a_beep_e_start, QA_a_beep_e_dur],
                      'break':[QA_break_start, QA_break_dur],
                      'censor_word':[0, 0]}

    # Initialize components for Routine "Pause"
    Pause_str = 'Pause: Please have some rest and \n'+ continue_str
    Pause_comp_list = [
        audio_generator(name='audio', loc=audio_root+'block_finish.wav', secs=-1),
        key_resp_generator(name='key_resp'),
        textstim_generator(win=win, name='cont', content=Pause_str, pos=annot_pos)
        ]
    Pause = routine_init('Pause', Pause_comp_list)
    Pause['time'] = {'audio': [0, 12], 'key_resp':[12, None], 'cont':[0, None]}

    # Initialize components for Routine "Cali_de_post_intro"
    Cali_de_post_intro_text_str = '  - Task: Read out the displayed German sentence   \n' + \
        '  - Melodious beep sound: Trial start              \n' + \
        '  - Beep sound with increasing pitch: Rec. start   \n' + \
        '  - Beep sound with decreasing pitch: Rec. finish  \n'
    Cali_de_post_intro_comp_list = [
        textstim_generator(win=win, name='title', content='READ OUT BLOCK (GERMAN)', pos=title_pos),
        textstim_generator(win=win, name='text', content=Cali_de_post_intro_text_str, pos=text_pos),
        audio_generator(name='audio', loc=audio_root+'calibration/calibration.wav', secs=-1),
        key_resp_generator(name='key_resp'),
        textstim_generator(win=win, name='cont', content=continue_str, pos=annot_pos)
        ]
    Cali_de_post_intro = routine_init('Cali_de_post_intro', Cali_de_post_intro_comp_list)
    Cali_de_post_intro['time'] = {'title':[cali_intro_start, cali_intro_dur],
                                  'text':[cali_intro_start, cali_intro_dur],
                                  'audio':[cali_intro_start, cali_intro_dur],
                                  'key_resp':[cali_intro_cont_start, cali_intro_cont_dur],
                                  'cont':[cali_intro_cont_start, cali_intro_cont_dur]}

    # Initialize components for Routine "Cali_de_post_rec"
    Cali_de_post_rec_text_str = ''
    Cali_de_post_rec_comp_list = [
        textstim_generator(win=win, name='text', content=Cali_de_post_rec_text_str, pos=title_pos),
        audio_generator(name='beep_hint', loc=audio_root+'calibration/reminder.wav', secs=0.6),
        textstim_generator(win=win, name='question_text', content=Cali_de_pre_rec_text_str, pos=text_pos),
        audio_generator(name='beep_start', loc=audio_root+'calibration/C3A_C4A_tone_decrease_1s_new.wav', secs=1),
        rec_generator(name='recording', sps=fs, loc='./data/', n_rec_chn=n_rec_chn),
        audio_generator(name='beep_end', loc=audio_root+'calibration/C4A_C3A_tone_decrease_1s_new.wav', secs=1),
        textstim_generator(win=win, name='break', content='Short break', pos=instruction_annot_pos)
        ]

    Cali_de_post_rec = routine_init('Cali_de_post_rec', Cali_de_post_rec_comp_list)
    Cali_de_post_rec['time'] = {'text':[0, cali_text_dur],
                                'beep_hint':[cali_hint_start, cali_hint_dur],
                                'question_text':[cali_q_start, cali_q_dur],
                                'beep_start':[cali_a_beep_s_start, cali_a_beep_s_dur],
                                'recording':[cali_rec_start, cali_rec_dur],
                                'beep_end':[cali_a_beep_e_start, cali_a_beep_e_dur],
                                'break':[cali_break_start, cali_break_dur]}

    # Initialize components for Routine "the_end"
    the_end_comp_list = [
        audio_generator(name='audio', loc=audio_root+'exp_finish.wav', secs=-1),
        textstim_generator(win=win, name='text', content=' This experiment is finished. \nThanks for your participation.', pos=[0.7, 0.0])
        ]
    the_end = routine_init('the_end', the_end_comp_list)
    the_end['time'] = {'audio': [0, 10],  'text':[0, 10]}

    # Create some handy timers
    globalClock = core.Clock()  # to track the time since experiment started
    routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine

# -----------------------------------------------------------------------------------
# ------------------------------ Start Exp. -----------------------------------------
# -----------------------------------------------------------------------------------
# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expInfo['expName'], version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='nibs_minimal_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)

def breakpoint_logger(comp, value, run, block, trial):
    jxu_logging.info('%s %d %s %s %s' %(comp, value, run, block, trial) )
# ---------------------------------------------------
# ----------------- instruction ---------------------
# ---------------------------------------------------




if Instruction_flag:
    print('Log: instruction start')

    breakpoint_logger(comp='Instruction', value=1, run=None, block=None, trial=None)

    # ------Prepare to start Routine "instruction"-------
    # update component parameters for each repeat
    instruction['key_resp'].keys = []
    instruction['key_resp'].rt = []
    # keep track of which components have finished
    win, instruction, instructionComponents, t, frameN, continueRoutine = pre_run_comp(win, instruction)
    trigger_mat = np.zeros((len(instructionComponents) - 1, 2))
    comp_list = np.asarray([*instruction['time'].keys()])
    # trigger_encoding_sending('instruction', input_run=0, input_block=0, intro_rec=0, input_event=0)
    # -------Run Routine "instruction"-------
    while continueRoutine:
        # get current time
        frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
            instruction["clock"], win, frameN)

        # *instruction["text"]* updates
        win, instruction['text'], trigger_mat[0] = run_comp(
            win, instruction['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=instruction['time']['text'][0], duration=instruction['time']['text'][1])

        # *instruction["audio"]* updates
        win, instruction['audio'], trigger_mat[1] = run_comp(
            win, instruction['audio'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=instruction['time']['audio'][0], duration=instruction['time']['audio'][1])
        # *instruction['key_resp']* updates
        waitOnFlip=False
        win, instruction['key_resp'], continueRoutine, endExpNow, trigger_mat[2] = run_comp(
            win, instruction['key_resp'], 'key_resp', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=instruction['time']['key_resp'][0], duration=instruction['time']['key_resp'][1],
            waitOnFlip=waitOnFlip)

        # *instruction['cont']* updates
        win, instruction['cont'], trigger_mat[3] = run_comp(
            win, instruction['cont'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=instruction['time']['cont'][0], duration=instruction['time']['cont'][1])

        break_flag = False
        win, continueRoutine, break_flag = continue_justification(
            win, endExpNow, defaultKeyboard, continueRoutine, instructionComponents)

        if trigger_mat.sum(axis=0)[0]:
            pass # trigger_encoding_sending('instruction', input_run=0, input_block=0, intro_rec=0, input_event=trigger_mat)
        if break_flag:
            break
    # trigger_encoding_sending('instruction', input_run=0, input_block=0, intro_rec=0, input_event=2)
    # -------Ending Routine "instruction"-------
    for thisComponent in instructionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)

    thisExp = data_writer(thisExp, instruction, 'instruction', ['text', 'cont'])
    print('Log: instruction finish')
    breakpoint_logger(comp='Instruction', value=0, run=None, block=None, trial=None)
    routineTimer.reset()




# ---------------------------------------------------
# ---------------- Artifact_intro -------------------
# ---------------------------------------------------

if Artifact_intro_flag:
    artifact_action_cnt = -1
    print('Log: artifact_intro start')
    breakpoint_logger(comp='Artifact_intro', value=1, run=None, block=None, trial=None)
    # ------Prepare to start Routine "Cali_de_pre_intro"-------
    # update component parameters for each repeat
    Artifact_intro['key_resp'].keys = []
    Artifact_intro['key_resp'].rt = []
    Artifact_intro['audio'].setSound(audio_root+'artifact/intro.wav',secs=-1, hamming=True)

    # keep track of which components have finished
    win, Artifact_intro, Artifact_introComponents, t, frameN, continueRoutine = pre_run_comp(win, Artifact_intro)
    trigger_mat = np.zeros((len(Artifact_introComponents) - 1, 2))
    comp_list = np.asarray([*Artifact_intro['time'].keys()])

    # -------Run Routine "Artifact_intro"-------

    trigger_sending(event_dict['Arti_intro'][0], default_sleep=True) # Sending trigger 70 (Artifact_intro Start)
    while continueRoutine:
        # get current time
        frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
            Artifact_intro["clock"], win, frameN)
        # update/draw components on each frame

        # *Artifact_intro["title"]* updates
        win, Artifact_intro['title'], trigger_mat[0] = run_comp(
            win, Artifact_intro['title'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Artifact_intro['time']['title'][0], duration=Artifact_intro['time']['title'][1])
        # *Artifact_intro["title"]* updates
        win, Artifact_intro['text'], trigger_mat[1] = run_comp(
            win, Artifact_intro['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Artifact_intro['time']['text'][0], duration=Artifact_intro['time']['text'][1])
        # *Artifact_intro["audio"]* updates
        win, Artifact_intro['audio'], trigger_mat[2] = run_comp(
            win, Artifact_intro['audio'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Artifact_intro['time']['audio'][0], duration=Artifact_intro['time']['audio'][1])

        # *Artifact_intro['key_resp']* updates
        waitOnFlip=False
        win, Artifact_intro['key_resp'], continueRoutine, endExpNow, trigger_mat[3] = run_comp(
            win, Artifact_intro['key_resp'], 'key_resp', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Artifact_intro['time']['key_resp'][0], duration=Artifact_intro['time']['key_resp'][1],
            waitOnFlip=waitOnFlip)
        # *Artifact_intro['cont']* updates
        win, Artifact_intro['cont'], trigger_mat[4] = run_comp(
            win, Artifact_intro['cont'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Artifact_intro['time']['cont'][0], duration=Artifact_intro['time']['cont'][1],
            repeat_per_frame=True, repeat_content=continue_str)

        win, continueRoutine, break_flag = continue_justification(
            win, endExpNow, defaultKeyboard, continueRoutine, Artifact_introComponents)
        if trigger_mat.sum(axis=0)[0]:
            pass # trigger_encoding_sending('Calibration', input_run=0, input_block=0, intro_rec=0, input_event=trigger_mat)
        if break_flag:
            break

    # -------Ending Routine "Artifact_intro"-------
    for thisComponent in Artifact_introComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)

    thisExp = data_writer(thisExp, Artifact_intro, 'Artifact_intro', ['title', 'text', 'audio', 'cont'])

    trigger_sending(event_dict['Arti_intro'][1], default_sleep=True) # Sending trigger 71 (Artifact_intro End)
    print('Log: artifact_intro finish')
    breakpoint_logger(comp='Artifact_intro', value=0, run=None, block=None, trial=None)
    routineTimer.reset()



# ---------------------------------------------------
# ----------------- Artifact_rec --------------------
# ---------------------------------------------------


if Artifact_rec_flag:

    # ---------------------------------------------------------------------------
    # ------------------------ Start Artifact_rec Trial --------------------------
    # ---------------------------------------------------------------------------
    # set up handler to look after randomisation of conditions etc

    artifact_trial = data.TrialHandler(nReps=n_arti_trial, method='random',
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='artifact_trial')
    thisExp.addLoop(artifact_trial)  # add the loop to the experiment
    thisArtifact_trial = artifact_trial.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisCali_pre_trial.rgb)
    if thisArtifact_trial != None:
        for paramName in thisArtifact_trial:
            exec('{} = thisArtifact_trial[paramName]'.format(paramName))

    for thisArtifact_trial in artifact_trial:

        currentLoop = artifact_trial
        # abbreviate parameter names if possible (e.g. rgb = thisArtifact_trial.rgb)
        if thisArtifact_trial != None:
            for paramName in thisArtifact_trial:
                exec('{} = thisArtifact_trial[paramName]'.format(paramName))

        # ------Prepare to start Routine "Artifact_rec"-------
        print('Log: artifact_rec start: Trial ' + str(artifact_trial.thisN))
        breakpoint_logger(comp='Artifact_rec', value=1, run=None, block=None, trial=artifact_trial.thisN)

        artifact_action_cnt += 1
        Artifact_rec['action'].setSound(action_path_list[artifact_action_cnt], secs=-1, hamming=True)

        # keep track of which components have finished
        win, Artifact_rec, Artifact_recComponents, t, frameN, continueRoutine = pre_run_comp(win, Artifact_rec)
        trigger_mat = np.zeros((len(Artifact_recComponents) - 1, 2))
        comp_list = np.asarray([*Artifact_rec['time'].keys()])

        # -------Run Routine "Artifact_rec"-------
        trigger_sending(event_dict['Arti_trial'][0], default_sleep=True) # Sending trigger 72 (Artifact_trial Start)
        routineTimer.reset()
        routineTimer.add(artifact_total_time)
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                Artifact_rec["clock"], win, frameN)

            # *Artifact_rec["text"]* updates
            win, Artifact_rec['text'], trigger_mat[0] = run_comp(
                win, Artifact_rec['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Artifact_rec['time']['text'][0], duration=Artifact_rec['time']['text'][1])
            # *Artifact_rec["beep_hint"]* updates
            win, Artifact_rec['beep_hint'], trigger_mat[1] = run_comp(
                win, Artifact_rec['beep_hint'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Artifact_rec['time']['beep_hint'][0], duration=Artifact_rec['time']['beep_hint'][1])
            # *Artifact_rec["action"]* updates
            win, Artifact_rec['action'], trigger_mat[2] = run_comp(
                win, Artifact_rec['action'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Artifact_rec['time']['action'][0], duration=Artifact_rec['time']['action'][1])

            # *Artifact_rec["beep_start"]* updates
            win, Artifact_rec['beep_start'], trigger_mat[3] = run_comp(
                win, Artifact_rec['beep_start'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Artifact_rec['time']['beep_start'][0], duration=Artifact_rec['time']['beep_start'][1])
            # *Artifact_rec["recording"]* updates
            win, Artifact_rec['recording'], trigger_mat[4] = run_comp(
                win, Artifact_rec['recording'], 'recording', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Artifact_rec['time']['recording'][0], duration=Artifact_rec['time']['recording'][1])
            # *Artifact_rec["beep_end"]* updates
            win, Artifact_rec['beep_end'], trigger_mat[5] = run_comp(
                win, Artifact_rec['beep_end'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Artifact_rec['time']['beep_end'][0], duration=Artifact_rec['time']['beep_end'][1])

            win, Artifact_rec['break'], trigger_mat[6] = run_comp(
                win, Artifact_rec['break'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Artifact_rec['time']['break'][0], duration=Artifact_rec['time']['break'][1])

            win, continueRoutine, break_flag = continue_justification(
                win, endExpNow, defaultKeyboard, continueRoutine, Artifact_recComponents)

            if trigger_mat.sum(axis=0)[0]:
                trigger_encoding_sending('Artifact', input_event=trigger_mat)
            if break_flag:
                break
        # -------Ending Routine "Artifact_rec"-------
        for thisComponent in Artifact_recComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)

        thisExp = data_writer(thisExp, Artifact_rec, 'Artifact_rec', ['text', 'beep_hint', 'action', 'beep_start', 'recording', 'beep_end', 'break'])
        thisExp.nextEntry()

        trigger_sending(event_dict['Arti_trial'][1], default_sleep=True) # Sending trigger 73 (Artifact_trial End)
        print('Log: artifact_rec finish: Trial' + str(artifact_trial.thisN))
        breakpoint_logger(comp='Artifact_rec', value=0, run=None, block=None, trial=artifact_trial.thisN)
        # the Routine "Artifact_rec" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()


# ---------------------------------------------------
# -------------- Cali_de_pre_intro ------------------
# ---------------------------------------------------

trigger_sending(event_dict['Pre_run'][0], default_sleep=True) # Sending trigger 2 (Pre-Run Start)

if Cali_de_pre_intro_flag:
    print('Log: cali_pre_intro start')
    breakpoint_logger(comp='Cali_de_pre_intro', value=1, run=None, block=None, trial=None)
    # ------Prepare to start Routine "Cali_de_pre_intro"-------
    # update component parameters for each repeat
    Cali_de_pre_intro['key_resp'].keys = []
    Cali_de_pre_intro['key_resp'].rt = []
    Cali_de_pre_intro['audio'].setSound(audio_root+'calibration/calibration.wav',secs=-1, hamming=True)

    # keep track of which components have finished
    win, Cali_de_pre_intro, Cali_de_pre_introComponents, t, frameN, continueRoutine = pre_run_comp(win, Cali_de_pre_intro)
    trigger_mat = np.zeros((len(Cali_de_pre_introComponents) - 1, 2))
    comp_list = np.asarray([*Cali_de_pre_intro['time'].keys()])

    # -------Run Routine "Cali_de_pre_intro"-------

    trigger_sending(event_dict['Cali_intro'][0], default_sleep=True) # Sending trigger 10 (Cali_de_pre_intro Start)
    while continueRoutine:
        # get current time
        frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
            Cali_de_pre_intro["clock"], win, frameN)
        # update/draw components on each frame

        # *Cali_de_pre_intro["title"]* updates
        win, Cali_de_pre_intro['title'], trigger_mat[0] = run_comp(
            win, Cali_de_pre_intro['title'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Cali_de_pre_intro['time']['title'][0], duration=Cali_de_pre_intro['time']['title'][1])
        # *Cali_de_pre_intro["title"]* updates
        win, Cali_de_pre_intro['text'], trigger_mat[1] = run_comp(
            win, Cali_de_pre_intro['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Cali_de_pre_intro['time']['text'][0], duration=Cali_de_pre_intro['time']['text'][1])
        # *Cali_de_pre_intro["audio"]* updates
        win, Cali_de_pre_intro['audio'], trigger_mat[2] = run_comp(
            win, Cali_de_pre_intro['audio'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Cali_de_pre_intro['time']['audio'][0], duration=Cali_de_pre_intro['time']['audio'][1])

        # *Cali_de_pre_intro['key_resp']* updates
        waitOnFlip=False
        win, Cali_de_pre_intro['key_resp'], continueRoutine, endExpNow, trigger_mat[3] = run_comp(
            win, Cali_de_pre_intro['key_resp'], 'key_resp', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Cali_de_pre_intro['time']['key_resp'][0], duration=Cali_de_pre_intro['time']['key_resp'][1],
            waitOnFlip=waitOnFlip)
        # *Cali_de_pre_intro['cont']* updates
        win, Cali_de_pre_intro['cont'], trigger_mat[4] = run_comp(
            win, Cali_de_pre_intro['cont'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Cali_de_pre_intro['time']['cont'][0], duration=Cali_de_pre_intro['time']['cont'][1],
            repeat_per_frame=True, repeat_content=continue_str)

        win, continueRoutine, break_flag = continue_justification(
            win, endExpNow, defaultKeyboard, continueRoutine, Cali_de_pre_introComponents)
        if trigger_mat.sum(axis=0)[0]:
            pass # trigger_encoding_sending('Calibration', input_run=0, input_block=0, intro_rec=0, input_event=trigger_mat)
        if break_flag:
            break

    # -------Ending Routine "Cali_de_pre_intro"-------
    for thisComponent in Cali_de_pre_introComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)

    thisExp = data_writer(thisExp, Cali_de_pre_intro, 'Cali_de_pre_intro', ['title', 'text', 'audio', 'cont'])

    trigger_sending(event_dict['Cali_intro'][1], default_sleep=True) # Sending trigger 11 (Cali_de_pre_intro End)
    print('Log: cali_pre_intro finish')
    breakpoint_logger(comp='Cali_de_pre_intro', value=0, run=None, block=None, trial=None)
    routineTimer.reset()

# ---------------------------------------------------
# -------------- Cali_de_pre_rec ------------------
# ---------------------------------------------------


if Cali_de_pre_rec_flag:

    # ---------------------------------------------------------------------------
    # ------------------------ Start Calibration Trial --------------------------
    # ---------------------------------------------------------------------------
    # set up handler to look after randomisation of conditions etc

    cali_pre_trial = data.TrialHandler(nReps=n_cali_trial, method='random',
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='pre_trial')
    thisExp.addLoop(cali_pre_trial)  # add the loop to the experiment
    thisCali_pre_trial = cali_pre_trial.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisCali_pre_trial.rgb)
    if thisCali_pre_trial != None:
        for paramName in thisCali_pre_trial:
            exec('{} = thisCali_pre_trial[paramName]'.format(paramName))

    for thisCali_pre_trial in cali_pre_trial:

        if break_cali_pre_trial != None and cali_pre_trial.thisN < break_cali_pre_trial:
            cali_question_cnt += 1
            continue
        break_cali_pre_trial = None   # clear the breakpoint

        currentLoop = cali_pre_trial
        # abbreviate parameter names if possible (e.g. rgb = thisCali_pre_trial.rgb)
        if thisCali_pre_trial != None:
            for paramName in thisCali_pre_trial:
                exec('{} = thisCali_pre_trial[paramName]'.format(paramName))

        # ------Prepare to start Routine "Cali_de_pre_rec"-------
        print('Log: cali_pre_rec start: Trial ' + str(cali_pre_trial.thisN))
        breakpoint_logger(comp='Cali_de_pre_rec', value=1, run=None, block=None, trial=cali_pre_trial.thisN)

        if external_question_flag:
            cali_question_cnt += 1
            Cali_de_pre_rec['question_text'].setText(cali_sen_text[cali_question_cnt])
        else:
            Cali_de_pre_rec['question_text'].setText('Text ' + str(cali_pre_trial.thisN))
        # keep track of which components have finished
        win, Cali_de_pre_rec, Cali_de_pre_recComponents, t, frameN, continueRoutine = pre_run_comp(win, Cali_de_pre_rec)
        trigger_mat = np.zeros((len(Cali_de_pre_recComponents) - 1, 2))
        comp_list = np.asarray([*Cali_de_pre_rec['time'].keys()])

        # -------Run Routine "Cali_de_pre_rec"-------
        trigger_sending(event_dict['Cali_trial'][0], default_sleep=True) # Sending trigger 12 (Cali_trial Start)
        routineTimer.reset()
        routineTimer.add(cali_total_time)
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                Cali_de_pre_rec["clock"], win, frameN)

            # *Cali_de_pre_rec["text"]* updates
            win, Cali_de_pre_rec['text'], trigger_mat[0] = run_comp(
                win, Cali_de_pre_rec['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Cali_de_pre_rec['time']['text'][0], duration=Cali_de_pre_rec['time']['text'][1])
            # *Cali_de_pre_rec["beep_hint"]* updates
            win, Cali_de_pre_rec['beep_hint'], trigger_mat[1] = run_comp(
                win, Cali_de_pre_rec['beep_hint'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Cali_de_pre_rec['time']['beep_hint'][0], duration=Cali_de_pre_rec['time']['beep_hint'][1])

            # *Cali_de_pre_rec["question_text"]* updates
            win, Cali_de_pre_rec['question_text'], trigger_mat[2] = run_comp(
                win, Cali_de_pre_rec['question_text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Cali_de_pre_rec['time']['question_text'][0], duration=Cali_de_pre_rec['time']['question_text'][1])

            # *Cali_de_pre_rec["beep_start"]* updates
            win, Cali_de_pre_rec['beep_start'], trigger_mat[3] = run_comp(
                win, Cali_de_pre_rec['beep_start'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Cali_de_pre_rec['time']['beep_start'][0], duration=Cali_de_pre_rec['time']['beep_start'][1])
            # *Cali_de_pre_rec["recording"]* updates
            win, Cali_de_pre_rec['recording'], trigger_mat[4] = run_comp(
                win, Cali_de_pre_rec['recording'], 'recording', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Cali_de_pre_rec['time']['recording'][0], duration=Cali_de_pre_rec['time']['recording'][1])
            # *Cali_de_pre_rec["beep_end"]* updates
            win, Cali_de_pre_rec['beep_end'], trigger_mat[5] = run_comp(
                win, Cali_de_pre_rec['beep_end'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Cali_de_pre_rec['time']['beep_end'][0], duration=Cali_de_pre_rec['time']['beep_end'][1])

            win, Cali_de_pre_rec['break'], trigger_mat[6] = run_comp(
                win, Cali_de_pre_rec['break'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Cali_de_pre_rec['time']['break'][0], duration=Cali_de_pre_rec['time']['break'][1])

            win, continueRoutine, break_flag = continue_justification(
                win, endExpNow, defaultKeyboard, continueRoutine, Cali_de_pre_recComponents)

            if trigger_mat.sum(axis=0)[0]:
                trigger_encoding_sending('Calibration', input_event=trigger_mat)
            if break_flag:
                break
        # trigger_encoding_sending('Calibration', input_run=0, input_block=0, intro_rec=1, input_event=6)
        # -------Ending Routine "Cali_de_pre_rec"-------
        for thisComponent in Cali_de_pre_recComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)

        thisExp = data_writer(thisExp, Cali_de_pre_rec, 'Cali_de_pre_rec', ['text', 'beep_hint', 'question_text', 'beep_start', 'beep_end', 'break'])

        # cali_pre_rec_file = folder_path + 'rec_cali_de_pre_' + + ' .wav'
        cali_pre_rec_file = folder_path + 'rec_cali_de_pre_' + 'trial_' + str(cali_pre_trial.thisN).zfill(3)  + '.wav'
        write(cali_pre_rec_file, fs, Cali_de_pre_rec['recording'].file)  # Save as WAV file
        print('Recording is saved!' + cali_pre_rec_file)
        # Add the detected time into the PsychoPy data file:
        thisExp.addData('filename', cali_pre_rec_file)
        thisExp.nextEntry()

        trigger_sending(event_dict['Cali_trial'][1], default_sleep=True) # Sending trigger 13 (Cali_trial End)
        print('Log: cali_pre_rec finish: Trial' + str(cali_pre_trial.thisN))
        breakpoint_logger(comp='Cali_de_pre_rec', value=0, run=None, block=None, trial=cali_pre_trial.thisN)
        # the Routine "Cali_de_pre_rec" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()


    # ---------------------------------------------------
    # --------------------- Pause -----------------------
    # ---------------------------------------------------
    if Pause_flag:

        trigger_sending(event_dict['Pause'][0], default_sleep=True) # Sending trigger 60 (Pause Start)
        # ------Prepare to start Routine "Pause"-------
        # update component parameters for each repeat
        Pause['key_resp'].keys = []
        Pause['key_resp'].rt = []
        Pause['cont'].setText('Pause: Please have some rest and '+ continue_str)

        # keep track of which components have finished
        win, Pause, PauseComponents, t, frameN, continueRoutine = pre_run_comp(win, Pause)
        # -------Run Routine "Pause"-------
        while continueRoutine:
            # get current time
            frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                Pause["clock"], win, frameN)
            # update/draw components on each frame
            # *Pause['audio']* updates
            win, Pause['audio'], trigger = run_comp(
                win, Pause['audio'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Pause['time']['audio'][0], duration=Pause['time']['audio'][1])

            waitOnFlip=False
            win, Pause['key_resp'], continueRoutine, endExpNow, trigger = run_comp(
                win, Pause['key_resp'], 'key_resp', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Pause['time']['key_resp'][0], duration=Pause['time']['key_resp'][1],
                waitOnFlip=waitOnFlip)
            # *Pause['cont']* updates
            win, Pause['cont'], trigger = run_comp(
                win, Pause['cont'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Pause['time']['cont'][0], duration=Pause['time']['cont'][1],
                repeat_per_frame=True, repeat_content='Please have some rest and press space key to continue')

            win, continueRoutine, break_flag = continue_justification(
                win, endExpNow, defaultKeyboard, continueRoutine, PauseComponents)
            if break_flag:
                break
        # -------Ending Routine "Pause"-------
        for thisComponent in PauseComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Pause_cont.started', Pause['cont'].tStartRefresh)
        thisExp.addData('Pause_cont.stopped', Pause['cont'].tStopRefresh)

        thisExp = data_writer(thisExp, Pause, 'Pause', ['cont'])
        trigger_sending(event_dict['Pause'][1], default_sleep=True) # Sending trigger 61 (Pause End)
        # the Routine "Pause" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()




cali_question_cnt = max(9, cali_question_cnt) # In case rerun, no reuse data

trigger_sending(event_dict['Pre_run'][1], default_sleep=True) # Sending trigger 3 (Pre-Run End)

# -----------------------------------------------------------------------------------
# ------------------------------ Start Run ------------------------------------------
# -----------------------------------------------------------------------------------


# set up handler to look after randomisation of conditions etc
run = data.TrialHandler(nReps=n_run, method='random',
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='run')
thisExp.addLoop(run)  # add the loop to the experiment
thisRun = run.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
if thisRun != None:
    for paramName in thisRun:
        exec('{} = thisRun[paramName]'.format(paramName))

for thisRun in run:

    if break_run != None and run.thisN < break_run:
        qa_question_cnt += n_trial * n_block  # In case rerun
        continue
    break_run = None   # clear the breakpoint

    trigger_sending(event_dict['Run'][0], default_sleep=True) # Sending trigger 4 (Run Start)
    currentLoop = run

    # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
    if thisRun != None:
        for paramName in thisRun:
            exec('{} = thisRun[paramName]'.format(paramName))

    # ---------------------------------------------------
    # --------------------- fade in ---------------------
    # ---------------------------------------------------
    if fade_in_flag:
        breakpoint_logger(comp='fade_in', value=1, run=run.thisN, block=None, trial=None)
        print('Log: fade in start: Run ' + str(run.thisN))

        fade_in['text'].setText('Please remain seated until further instruction via the earphone.')
        # Instantiate the object for signal generator
        try:
            if isinstance(fg, SG()):
                pass
        except:
            fg = SG()

        # Fade in trigger: If sham stim, send trigger 22, otherwise 24
        if run.thisN == stim_run[0]:
            intensity_change_flag = 'i'
            if stim_freq == 0:
                trigger_sending(event_dict['Sham'][0], default_sleep=True) # Sending trigger 22 (Sham stim Start)
                win.flip()
                time.sleep(2.000)
            else:
                trigger_sending(event_dict['Stim'][0], default_sleep=True) # Sending trigger 20 (Real stim Start)
                fg.amp(init_intensity)
                fg.frequency(stim_freq)
                time.sleep(1.0)
                fg.on()
                time.sleep(1.0)
        else:
            intensity_change_flag = 'keep'
        # if np.abs(fg.get_amp()- max) > 0.01 and run.thisN in stim_run:
        #     intensity_change_flag = 'i'


        # ------Prepare to start Routine "fade_in"-------
        # keep track of which components have finished



        # To be able to enter to loop of maintain/update intensity
        tmp_intensity = None
        if not input_intensity < max_intensity:
            tmp_intensity = input_intensity
            input_intensity = max_intensity - 0.05

        if intensity_change_flag == 'i':
            trigger_sending(event_dict['Fade_in'][0], default_sleep=True) # Sending trigger 24 (Fade_in Start)
        fade_itr_cnt = 0
        while input_intensity < max_intensity and fade_itr_cnt < n_step_fade_stim:
            if tmp_intensity != None:
                input_intensity = tmp_intensity
                tmp_intensity = None
                print('initial ' + str(input_intensity))

            win, fade_in, fade_inComponents, t, frameN, continueRoutine = pre_run_comp(win, fade_in)
            trigger_mat = np.zeros((len(fade_inComponents) - 1, 2))
            comp_list = np.asarray([*fade_in['time'].keys()])

            routineTimer.reset()
            routineTimer.add(fade_in['time']['auto_stim'][1])

            # -------Run Routine "fade_in"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                    fade_in["clock"], win, frameN)
                # *fade_in["text"]* updates
                win, fade_in['text'], trigger_mat[0] = run_comp(
                    win, fade_in['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                    start_time=fade_in['time']['text'][0], duration=fade_in['time']['text'][1])

                # Only maintain or update the stim intensity during stim run
                if stim_freq != 0 and run.thisN == stim_run[0]:
                    win, fade_in['auto_stim'], output_intensity, stim_continue, continueRoutine, endExpNow, intensity_change_flag, trigger_mat[1] = run_comp(
                        win, fade_in['auto_stim'], 'auto_stim', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=fade_in['time']['auto_stim'][0], duration=fade_in['time']['auto_stim'][1],
                        stim_current_intensity=input_intensity, stim_intensity_limit=[min_intensity, max_intensity],
                        stim_step_intensity=step_intensity, stim_obj=fg, intensity_change_flag=intensity_change_flag,
                        stim=True)
                    input_intensity = output_intensity

                break_flag = False
                win, continueRoutine, break_flag = continue_justification(
                    win, endExpNow, defaultKeyboard, continueRoutine, fade_inComponents)

                if trigger_mat.sum(axis=0)[0]:
                    pass
                if break_flag:
                    break
            if stim_freq != 0 and run.thisN == stim_run[0]:
                intensity_change_flag = 'i'
            else:
                intensity_change_flag = 'keep'
            fade_itr_cnt += 1

        # -------Ending Routine "fade_in"-------
        for thisComponent in fade_inComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)

        thisExp = data_writer(thisExp, fade_in, 'fade_in', ['text'])
        if intensity_change_flag == 'i':
            trigger_sending(event_dict['Fade_in'][1], default_sleep=True) # Sending trigger 25 (Fade_in End)


        print('Log: fade in end: Run ' + str(run.thisN))
        breakpoint_logger(comp='fade_in', value=0, run=run.thisN, block=None, trial=None)
        routineTimer.reset()

    trigger_sending(event_dict['Stable_stim'][0], default_sleep=True) # Sending trigger 28 (Stable stim Start)


    if run.thisN == 2:
        artifact_action_cnt = -1
        # ---------------------------------------------------
        # ---------------- Artifact_intro -------------------
        # ---------------------------------------------------

        if Artifact_intro_within_flag:
            print('Log: artifact_intro start')
            breakpoint_logger(comp='Artifact_intro', value=1, run=None, block=None, trial=None)
            # ------Prepare to start Routine "Cali_de_pre_intro"-------
            # update component parameters for each repeat
            Artifact_intro['key_resp'].keys = []
            Artifact_intro['key_resp'].rt = []
            Artifact_intro['audio'].setSound(audio_root+'artifact/intro.wav',secs=-1, hamming=True)

            # keep track of which components have finished
            win, Artifact_intro, Artifact_introComponents, t, frameN, continueRoutine = pre_run_comp(win, Artifact_intro)
            trigger_mat = np.zeros((len(Artifact_introComponents) - 1, 2))
            comp_list = np.asarray([*Artifact_intro['time'].keys()])

            # -------Run Routine "Artifact_intro"-------

            trigger_sending(event_dict['Arti_intro'][0], default_sleep=True) # Sending trigger 70 (Artifact_intro Start)
            while continueRoutine:
                # get current time
                frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                    Artifact_intro["clock"], win, frameN)
                # update/draw components on each frame

                # *Artifact_intro["title"]* updates
                win, Artifact_intro['title'], trigger_mat[0] = run_comp(
                    win, Artifact_intro['title'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                    start_time=Artifact_intro['time']['title'][0], duration=Artifact_intro['time']['title'][1])
                # *Artifact_intro["title"]* updates
                win, Artifact_intro['text'], trigger_mat[1] = run_comp(
                    win, Artifact_intro['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                    start_time=Artifact_intro['time']['text'][0], duration=Artifact_intro['time']['text'][1])
                # *Artifact_intro["audio"]* updates
                win, Artifact_intro['audio'], trigger_mat[2] = run_comp(
                    win, Artifact_intro['audio'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                    start_time=Artifact_intro['time']['audio'][0], duration=Artifact_intro['time']['audio'][1])

                # *Artifact_intro['key_resp']* updates
                waitOnFlip=False
                win, Artifact_intro['key_resp'], continueRoutine, endExpNow, trigger_mat[3] = run_comp(
                    win, Artifact_intro['key_resp'], 'key_resp', frameN, t, tThisFlip, tThisFlipGlobal,
                    start_time=Artifact_intro['time']['key_resp'][0], duration=Artifact_intro['time']['key_resp'][1],
                    waitOnFlip=waitOnFlip)
                # *Artifact_intro['cont']* updates
                win, Artifact_intro['cont'], trigger_mat[4] = run_comp(
                    win, Artifact_intro['cont'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                    start_time=Artifact_intro['time']['cont'][0], duration=Artifact_intro['time']['cont'][1],
                    repeat_per_frame=True, repeat_content=continue_str)

                win, continueRoutine, break_flag = continue_justification(
                    win, endExpNow, defaultKeyboard, continueRoutine, Artifact_introComponents)
                if trigger_mat.sum(axis=0)[0]:
                    pass # trigger_encoding_sending('Calibration', input_run=0, input_block=0, intro_rec=0, input_event=trigger_mat)
                if break_flag:
                    break

            # -------Ending Routine "Artifact_intro"-------
            for thisComponent in Artifact_introComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)

            thisExp = data_writer(thisExp, Artifact_intro, 'Artifact_intro', ['title', 'text', 'audio', 'cont'])

            trigger_sending(event_dict['Arti_intro'][1], default_sleep=True) # Sending trigger 71 (Artifact_intro End)
            print('Log: artifact_intro finish')
            breakpoint_logger(comp='Artifact_intro', value=0, run=None, block=None, trial=None)
            routineTimer.reset()



        # ---------------------------------------------------
        # ----------------- Artifact_rec --------------------
        # ---------------------------------------------------


        if Artifact_rec_within_flag:

            # ---------------------------------------------------------------------------
            # ------------------------ Start Artifact_rec Trial --------------------------
            # ---------------------------------------------------------------------------
            # set up handler to look after randomisation of conditions etc

            artifact_trial = data.TrialHandler(nReps=n_arti_trial, method='random',
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='artifact_trial')
            thisExp.addLoop(artifact_trial)  # add the loop to the experiment
            thisArtifact_trial = artifact_trial.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisCali_pre_trial.rgb)
            if thisArtifact_trial != None:
                for paramName in thisArtifact_trial:
                    exec('{} = thisArtifact_trial[paramName]'.format(paramName))

            for thisArtifact_trial in artifact_trial:

                currentLoop = artifact_trial
                # abbreviate parameter names if possible (e.g. rgb = thisArtifact_trial.rgb)
                if thisArtifact_trial != None:
                    for paramName in thisArtifact_trial:
                        exec('{} = thisArtifact_trial[paramName]'.format(paramName))

                # ------Prepare to start Routine "Artifact_rec"-------
                print('Log: artifact_rec start: Trial ' + str(artifact_trial.thisN))
                breakpoint_logger(comp='Artifact_rec', value=1, run=None, block=None, trial=artifact_trial.thisN)

                artifact_action_cnt += 1
                Artifact_rec['action'].setSound(action_path_list[artifact_action_cnt], secs=-1, hamming=True)

                # keep track of which components have finished
                win, Artifact_rec, Artifact_recComponents, t, frameN, continueRoutine = pre_run_comp(win, Artifact_rec)
                trigger_mat = np.zeros((len(Artifact_recComponents) - 1, 2))
                comp_list = np.asarray([*Artifact_rec['time'].keys()])

                # -------Run Routine "Artifact_rec"-------
                trigger_sending(event_dict['Arti_trial'][0], default_sleep=True) # Sending trigger 72 (Artifact_trial Start)
                routineTimer.reset()
                routineTimer.add(artifact_total_time)
                while continueRoutine and routineTimer.getTime() > 0:
                    # get current time
                    frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                        Artifact_rec["clock"], win, frameN)

                    # *Artifact_rec["text"]* updates
                    win, Artifact_rec['text'], trigger_mat[0] = run_comp(
                        win, Artifact_rec['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=Artifact_rec['time']['text'][0], duration=Artifact_rec['time']['text'][1])
                    # *Artifact_rec["beep_hint"]* updates
                    win, Artifact_rec['beep_hint'], trigger_mat[1] = run_comp(
                        win, Artifact_rec['beep_hint'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=Artifact_rec['time']['beep_hint'][0], duration=Artifact_rec['time']['beep_hint'][1])
                    # *Artifact_rec["action"]* updates
                    win, Artifact_rec['action'], trigger_mat[2] = run_comp(
                        win, Artifact_rec['action'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=Artifact_rec['time']['action'][0], duration=Artifact_rec['time']['action'][1])

                    # *Artifact_rec["beep_start"]* updates
                    win, Artifact_rec['beep_start'], trigger_mat[3] = run_comp(
                        win, Artifact_rec['beep_start'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=Artifact_rec['time']['beep_start'][0], duration=Artifact_rec['time']['beep_start'][1])
                    # *Artifact_rec["recording"]* updates
                    win, Artifact_rec['recording'], trigger_mat[4] = run_comp(
                        win, Artifact_rec['recording'], 'recording', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=Artifact_rec['time']['recording'][0], duration=Artifact_rec['time']['recording'][1])
                    # *Artifact_rec["beep_end"]* updates
                    win, Artifact_rec['beep_end'], trigger_mat[5] = run_comp(
                        win, Artifact_rec['beep_end'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=Artifact_rec['time']['beep_end'][0], duration=Artifact_rec['time']['beep_end'][1])

                    win, Artifact_rec['break'], trigger_mat[6] = run_comp(
                        win, Artifact_rec['break'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=Artifact_rec['time']['break'][0], duration=Artifact_rec['time']['break'][1])

                    win, continueRoutine, break_flag = continue_justification(
                        win, endExpNow, defaultKeyboard, continueRoutine, Artifact_recComponents)

                    if trigger_mat.sum(axis=0)[0]:
                        trigger_encoding_sending('Artifact', input_event=trigger_mat)
                    if break_flag:
                        break
                # -------Ending Routine "Artifact_rec"-------
                for thisComponent in Artifact_recComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)

                thisExp = data_writer(thisExp, Artifact_rec, 'Artifact_rec', ['text', 'beep_hint', 'action', 'beep_start', 'recording', 'beep_end', 'break'])
                thisExp.nextEntry()

                trigger_sending(event_dict['Arti_trial'][1], default_sleep=True) # Sending trigger 73 (Artifact_trial End)
                print('Log: artifact_rec finish: Trial' + str(artifact_trial.thisN))
                breakpoint_logger(comp='Artifact_rec', value=0, run=None, block=None, trial=artifact_trial.thisN)
                # the Routine "Artifact_rec" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()


    # ---------------------------------------------------------
    # ---------------- Resting state block --------------------
    # ---------------------------------------------------------

    trigger_sending(event_dict['Block'][0], default_sleep=True) # Sending trigger 6 (Block Start)
    RS_order = ['open', 'close']
    random.shuffle(RS_order)
    for RS_loop in range(2):

        if break_rs_block != None and RS_loop < break_rs_block:
            continue
        break_rs_block = None   # clear the breakpoint

        # ---------------------------------------------------
        # ------------------- RS_intro ----------------------
        # ---------------------------------------------------

        if RS_intro_flag:
            print('Log: RS intro start: Run ' + str(run.thisN) + RS_order[RS_loop] + ' Block ' + ': ' + str(RS_loop))
            breakpoint_logger(comp='RS_intro', value=1, run=run.thisN, block=RS_loop, trial=None)
            # ------Prepare to start Routine "RS_intro"-------
            RS_intro['audio'].setSound(audio_root+'resting_state/rs_intro_' + RS_order[RS_loop] + '.wav', secs=-1, hamming=True)

            # keep track of which components have finished
            win, RS_intro, RS_introComponents, t, frameN, continueRoutine = pre_run_comp(win, RS_intro)
            trigger_mat = np.zeros((len(RS_introComponents) - 1, 2))
            comp_list = np.asarray([*RS_intro['time'].keys()])

            trigger_sending(event_dict['RS_intro'][0], default_sleep=True) # Sending trigger 30 (RS_intro Start)
            # -------Run Routine "RS_intro"-------
            while continueRoutine:
                # get current time
                frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                    RS_intro["clock"], win, frameN)

                # *RS_intro["title"]* updates
                win, RS_intro['title'], trigger_mat[0] = run_comp(
                    win, RS_intro['title'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                    start_time=RS_intro['time']['title'][0], duration=RS_intro['time']['title'][1])
                # *Cali_de_pre_intro["title"]* updates
                win, RS_intro['text'], trigger_mat[1] = run_comp(
                    win, RS_intro['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                    start_time=RS_intro['time']['text'][0], duration=RS_intro['time']['text'][1])
                # *Cali_de_pre_intro["audio"]* updates
                win, RS_intro['audio'], trigger_mat[2] = run_comp(
                    win, RS_intro['audio'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                    start_time=RS_intro['time']['audio'][0], duration=RS_intro['time']['audio'][1])


                win, continueRoutine, break_flag = continue_justification(
                    win, endExpNow, defaultKeyboard, continueRoutine, RS_introComponents)

                if trigger_mat.sum(axis=0)[0]:
                    pass
                if break_flag:
                    break

            # -------Ending Routine "RS_intro"-------
            for thisComponent in RS_introComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)

            run = data_writer(run, RS_intro, 'RS_intro', ['title', 'text', 'audio'])
            trigger_sending(event_dict['RS_intro'][1], default_sleep=True) # Sending trigger 31 (RS_intro Start)
            # the Routine "RS_intro" was not non-slip safe, so reset the non-slip timer
            print('Log: RS intro finish: Run ' + str(run.thisN) + RS_order[RS_loop] + ' Block ' + ': ' + str(RS_loop))
            breakpoint_logger(comp='RS_intro', value=0, run=run.thisN, block=RS_loop, trial=None)
            routineTimer.reset()

        # ---------------------------------------------------
        # --------------------- RS_rec ----------------------
        # ---------------------------------------------------
        if RS_rec_flag:
            print('Log: RS rec start: Run ' + str(run.thisN) + RS_order[RS_loop] + ' Block ' + ': ' + str(RS_loop))
            breakpoint_logger(comp='RS_rec', value=1, run=run.thisN, block=RS_loop, trial=None)
            if RS_order[RS_loop] == 'open':
                trigger_sending(event_dict['RS_open'][0], default_sleep=True) # Sending trigger 32 (RS_open Start)
                RS_rec_text_str = 'Please keep relaxed and open your eyes.\nNote: Blinking is allowed.'
                RS_rec['text'].setText(RS_rec_text_str)
            elif RS_order[RS_loop] == 'close':
                trigger_sending(event_dict['RS_close'][0], default_sleep=True) # Sending trigger 34 (RS_close Start)
                RS_rec_text_str = 'Please keep relaxed and close your eyes.\n'
                RS_rec['text'].setText(RS_rec_text_str)
            # ------Prepare to start Routine "RS_rec"-------
            # update component parameters for each repeat
            # keep track of which components have finished
            win, RS_rec, RS_recComponents, t, frameN, continueRoutine = pre_run_comp(win, RS_rec)
            trigger_mat = np.zeros((len(RS_recComponents) - 1, 2))
            comp_list = np.asarray([*RS_rec['time'].keys()])
            # trigger_encoding_sending('RS', input_run=run.thisRepN, input_block=0, intro_rec=1, input_event=0)
            # -------Run Routine "RS_rec"-------
            while continueRoutine:
                # get current time
                frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                    RS_rec["clock"], win, frameN)

                # *RS_rec["text"]* updates
                win, RS_rec['text'], trigger_mat[0] = run_comp(
                    win, RS_rec['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                    start_time=RS_rec['time']['text'][0], duration=RS_rec['time']['text'][1])
                # *QA_rec["beep_end"]* updates
                win, RS_rec['beep_end'], trigger_mat[1] = run_comp(
                    win, RS_rec['beep_end'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                    start_time=RS_rec['time']['beep_end'][0], duration=RS_rec['time']['beep_end'][1])

                win, continueRoutine, break_flag = continue_justification(
                    win, endExpNow, defaultKeyboard, continueRoutine, RS_recComponents)

                if trigger_mat.sum(axis=0)[0]:
                    pass # trigger_encoding_sending('RS', input_run=run.thisRepN, input_block=0, intro_rec=1, input_event=trigger_mat)
                if break_flag:
                    break
            # trigger_encoding_sending('RS', input_run=run.thisRepN, input_block=0, intro_rec=1, input_event=2)
            # -------Ending Routine "RS_rec"-------
            for thisComponent in RS_recComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            run = data_writer(run, RS_rec, 'RS_rec', ['text'])
            if RS_order[RS_loop] == 'open':
                trigger_sending(event_dict['RS_open'][1], default_sleep=True) # Sending trigger 34 (RS_open End)
            elif RS_order[RS_loop] == 'close':
                trigger_sending(event_dict['RS_close'][1], default_sleep=True) # Sending trigger 35 (RS_close End)
            # the Routine "RS_rec" was not non-slip safe, so reset the non-slip timer
            print('Log: RS rec finish: Run ' + str(run.thisN) + RS_order[RS_loop] + ' Block ' + ': ' + str(RS_loop))
            breakpoint_logger(comp='RS_rec', value=0, run=run.thisN, block=RS_order[RS_loop], trial=None)
            routineTimer.reset()

    # ---------------------------------------------------
    # --------------------- Pause -----------------------
    # ---------------------------------------------------
    if Pause_flag:

        # ------Prepare to start Routine "Pause"-------
        # update component parameters for each repeat
        Pause['key_resp'].keys = []
        Pause['key_resp'].rt = []
        Pause['cont'].setText('Press [space] key to continue.')
        # keep track of which components have finished
        win, Pause, PauseComponents, t, frameN, continueRoutine = pre_run_comp(win, Pause)
        # -------Run Routine "Pause"-------
        trigger_sending(event_dict['Pause'][0], default_sleep=True) # Sending trigger 60 (Pause Start)
        while continueRoutine:
            # get current time
            frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                Pause["clock"], win, frameN)
            # update/draw components on each frame
            # *Pause['audio']* updates
            win, Pause['audio'], trigger = run_comp(
                win, Pause['audio'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Pause['time']['audio'][0], duration=Pause['time']['audio'][1])

            waitOnFlip=False
            win, Pause['key_resp'], continueRoutine, endExpNow, trigger = run_comp(
                win, Pause['key_resp'], 'key_resp', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Pause['time']['key_resp'][0], duration=Pause['time']['key_resp'][1],
                waitOnFlip=waitOnFlip)
            # *Pause['cont']* updates
            win, Pause['cont'], trigger = run_comp(
                win, Pause['cont'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Pause['time']['cont'][0], duration=Pause['time']['cont'][1],
                repeat_per_frame=True, repeat_content='Please have some rest and press space key to continue')

            win, continueRoutine, break_flag = continue_justification(
                win, endExpNow, defaultKeyboard, continueRoutine, PauseComponents)
            if break_flag:
                break
        # -------Ending Routine "Pause"-------
        for thisComponent in PauseComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Pause_cont.started', Pause['cont'].tStartRefresh)
        thisExp.addData('Pause_cont.stopped', Pause['cont'].tStopRefresh)

        thisExp = data_writer(thisExp, Pause, 'Pause', ['cont'])
        trigger_sending(event_dict['Pause'][1], default_sleep=True) # Sending trigger 61 (Pause End)
        # the Routine "Pause" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()


    trigger_sending(event_dict['Block'][1], default_sleep=True) # Sending trigger 7 (Block End)

    # ---------------------------------------------------
    # ------------------- QA_intro ----------------------
    # ---------------------------------------------------
    # (ps. For QA_intro, it is unnecessary to repeat twice, hence no block trigger sent here)

    if QA_intro_flag:
        # ------Prepare to start Routine "QA_intro"-------
        # update component parameters for each repeat
        """
        QA_intro['audio'].setSound('../../../../Data/NIBS/Stage_one/Audio/Soundeffect/q_a/q_a.wav', secs=32, hamming=True)
        QA_intro['audio'].setVolume(1, log=False)
        """
        QA_intro['key_resp'].keys = []
        QA_intro['key_resp'].rt = []
        # keep track of which components have finished
        win, QA_intro, QA_introComponents, t, frameN, continueRoutine = pre_run_comp(win, QA_intro)
        trigger_mat = np.zeros((len(QA_introComponents) - 1, 2))
        comp_list = np.asarray([*QA_intro['time'].keys()])
        print('Log: QA intro start: Run ' + str(run.thisN))
        breakpoint_logger(comp='QA_intro', value=1, run=run.thisN, block=None, trial=None)
        # -------Run Routine "QA_intro"-------
        trigger_sending(event_dict['QA_intro'][0], default_sleep=True) # Sending trigger 40 (QA_intro Start)
        while continueRoutine:
            # get current time
            frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                QA_intro["clock"], win, frameN)
            # update/draw components on each frame

            # *RS_intro["title"]* updates
            win, QA_intro['title'], trigger_mat[0] = run_comp(
                win, QA_intro['title'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=QA_intro['time']['title'][0], duration=QA_intro['time']['title'][1])
            # *Cali_de_pre_intro["title"]* updates
            win, QA_intro['text'], trigger_mat[1] = run_comp(
                win, QA_intro['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=QA_intro['time']['text'][0], duration=QA_intro['time']['text'][1])
            # *Cali_de_pre_intro["audio"]* updates
            win, QA_intro['audio'], trigger_mat[2] = run_comp(
                win, QA_intro['audio'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=QA_intro['time']['audio'][0], duration=QA_intro['time']['audio'][1])

            # *Cali_de_pre_intro['key_resp']* updates
            waitOnFlip=False
            win, QA_intro['key_resp'], continueRoutine, endExpNow, trigger_mat[3] = run_comp(
                win, QA_intro['key_resp'], 'key_resp', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=QA_intro['time']['key_resp'][0], duration=QA_intro['time']['key_resp'][1],
                waitOnFlip=waitOnFlip)
            # *Cali_de_pre_intro['cont']* updates
            win, QA_intro['cont'], trigger_mat[4] = run_comp(
                win, QA_intro['cont'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=QA_intro['time']['cont'][0], duration=QA_intro['time']['cont'][1],
                repeat_per_frame=True, repeat_content=continue_str)

            win, continueRoutine, break_flag = continue_justification(
                win, endExpNow, defaultKeyboard, continueRoutine, QA_introComponents)

            if trigger_mat.sum(axis=0)[0]:
                pass
            if break_flag:
                break

        # -------Ending Routine "QA_intro"-------
        for thisComponent in QA_introComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        run = data_writer(run, QA_intro, 'QA_intro', ['title', 'text', 'audio', 'cont'])
        # the Routine "QA_intro" was not non-slip safe, so reset the non-slip timer
        print('Log: QA intro finish: Run ' + str(run.thisN))
        breakpoint_logger(comp='QA_intro', value=0, run=run.thisN, block=None, trial=None)
        routineTimer.reset()
        trigger_sending(event_dict['QA_intro'][1], default_sleep=True) # Sending trigger 41 (QA_intro End)
    # -------------------------------------------------------------------------------
    # ------------------------------ Start Block ------------------------------------
    # -------------------------------------------------------------------------------
    # set up handler to look after randomisation of conditions etc
    if isinstance(n_block, list):
        n_block_parsed = n_block[run.thisRepN]
    else:
        n_block_parsed = n_block

    QA_block = data.TrialHandler(nReps=n_block_parsed, method='random',
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='QA_block')
    thisExp.addLoop(QA_block)  # add the loop to the experiment
    thisQA_block = QA_block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisQA_block.rgb)
    if thisQA_block != None:
        for paramName in thisQA_block:
            exec('{} = thisQA_block[paramName]'.format(paramName))

    for thisQA_block in QA_block:

        if break_qa_block != None and QA_block.thisN < break_qa_block:
            qa_question_cnt += n_trial
            continue
        break_qa_block = None   # clear the breakpoint

        trigger_sending(event_dict['Block'][0], default_sleep=True) # Sending trigger 6 (Block Start)
        currentLoop = QA_block
        # abbreviate parameter names if possible (e.g. rgb = thisQA_block.rgb)
        if thisQA_block != None:
            for paramName in thisQA_block:
                exec('{} = thisQA_block[paramName]'.format(paramName))

        # ---------------------------------------------------------------------------
        # ------------------------------ Start Trial --------------------------------
        # ---------------------------------------------------------------------------
        # set up handler to look after randomisation of conditions etc

        QA_trial = data.TrialHandler(nReps=n_trial, method='random',
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='QA_trial')
        thisExp.addLoop(QA_trial)  # add the loop to the experiment
        thisQA_trial = QA_trial.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisQA_trial.rgb)
        if thisQA_trial != None:
            for paramName in thisQA_trial:
                exec('{} = thisQA_trial[paramName]'.format(paramName))

        for thisQA_trial in QA_trial:

            if break_qa_trial != None and QA_trial.thisN < break_qa_trial:
                qa_question_cnt += 1
                continue
            if break_qa_trial != None:
                if RS_intro_flag == 0:
                    RS_intro_flag = 1
                if RS_rec_flag == 0:
                    RS_rec_flag = 1
                if QA_intro_flag == 0:
                    QA_intro_flag = 1

            break_qa_trial = None   # clear the breakpoint
            currentLoop = QA_trial
            # abbreviate parameter names if possible (e.g. rgb = thisQA_trial.rgb)
            if thisQA_trial != None:
                for paramName in thisQA_trial:
                    exec('{} = thisQA_trial[paramName]'.format(paramName))

            # ---------------------------------------------------
            # -------------------- QA_rec -----------------------
            # ---------------------------------------------------

            if QA_rec_flag:
                print('Log: QA rec start: Run ' + str(run.thisN) + ' Block ' + str(QA_block.thisN) + 'Trial ' + str(QA_trial.thisN))
                breakpoint_logger(comp='QA_rec', value=1, run=run.thisN, block=QA_block.thisN, trial=QA_trial.thisN)

                # update component parameters for each repeat
                ques_start = QA_rec['time']['question'][0]
                if external_question_flag:
                    qa_question_cnt += 1

                    QA_rec['question'].setSound(question_path[qa_question_cnt], secs=sen_duration[qa_question_cnt] - 0.016, hamming=True)
                    QA_rec['time']['question'][1] = sen_duration[qa_question_cnt]  - 0.016  # 1 frame.
                    QA_rec['time']['censor_word'] = [ques_start + censor_question_start[qa_question_cnt], censor_question_duration[qa_question_cnt]]
                    QA_rec['question'].sen_text = sen_text[qa_question_cnt]
                    QA_rec['question'].cen_text = cen_text[qa_question_cnt]
                    if flexible_qa_rec_start:
                        QA_trial_dur = 30.00
                        QA_rec['time']['beep_start'][0] = QA_q_start + sen_duration[qa_question_cnt] + comp_gap
                        QA_rec['time']['recording'][0] = QA_rec['time']['beep_start'][0] + QA_a_beep_s_dur + comp_gap
                        QA_rec['time']['recording'][1] = QA_a_beep_e_start - QA_rec['time']['recording'][0] - comp_gap
                else:
                    QA_rec['question'].setSound('/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/old_data/article_0/sentence_0/sentence_0_syn_44100.wav', secs=-1, hamming=True)
                    QA_rec['time']['censor_word'] = [ques_start + 0.706, 0.694]
                    QA_rec['time']['question'][1] = 5.63 - 0.02
                # QA_rec['question'].setVolume(1, log=False)

                # keep track of which components have finished
                win, QA_rec, QA_recComponents, t, frameN, continueRoutine = pre_run_comp(win, QA_rec)
                trigger_mat = np.zeros((len(QA_recComponents) - 1, 2))
                comp_list = np.asarray([*QA_rec['time'].keys()])
                # ------Prepare to start Routine "QA_rec"-------
                routineTimer.reset()
                routineTimer.add(qa_total_time)
                trigger_sending(event_dict['QA_trial'][0], default_sleep=True) # Sending trigger 42 (QA_trial Start)
                # -------Run Routine "QA_rec"-------
                while continueRoutine and routineTimer.getTime() > 0:
                    # get current time
                    frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                        QA_rec["clock"], win, frameN)
                    # update/draw components on each frame

                    # *QA_rec["text"]* updates
                    win, QA_rec['text'], trigger_mat[0] = run_comp(
                        win, QA_rec['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=QA_rec['time']['text'][0], duration=QA_rec['time']['text'][1])
                    # *QA_rec["beep_hint"]* updates
                    win, QA_rec['beep_hint'], trigger_mat[1] = run_comp(
                        win, QA_rec['beep_hint'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=QA_rec['time']['beep_hint'][0], duration=QA_rec['time']['beep_hint'][1])
                    # *QA_rec["question"]* updates
                    win, QA_rec['question'], trigger_mat[2] = run_comp(
                        win, QA_rec['question'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=QA_rec['time']['question'][0], duration=QA_rec['time']['question'][1])
                    # *QA_rec["beep_start"]* updates
                    win, QA_rec['beep_start'], trigger_mat[3] = run_comp(
                        win, QA_rec['beep_start'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=QA_rec['time']['beep_start'][0], duration=QA_rec['time']['beep_start'][1])
                    # *QA_rec["recording"]* updates
                    win, QA_rec['recording'], trigger_mat[4] = run_comp(
                        win, QA_rec['recording'], 'recording', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=QA_rec['time']['recording'][0], duration=QA_rec['time']['recording'][1])
                    # *QA_rec["beep_end"]* updates
                    win, QA_rec['beep_end'], trigger_mat[5] = run_comp(
                        win, QA_rec['beep_end'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=QA_rec['time']['beep_end'][0], duration=QA_rec['time']['beep_end'][1])

                    win, QA_rec['break'], trigger_mat[6] = run_comp(
                        win, QA_rec['break'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=QA_rec['time']['break'][0], duration=QA_rec['time']['break'][1])
                    win, QA_rec['censor_word'],temp_del =run_comp(
                        win, QA_rec['censor_word'], 'trigger', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=QA_rec['time']['censor_word'][0], duration=QA_rec['time']['censor_word'][1])

                    win, continueRoutine, break_flag = continue_justification(
                        win, endExpNow, defaultKeyboard, continueRoutine, QA_recComponents)

                    if trigger_mat.sum(axis=0)[0]:
                        trigger_encoding_sending('QA', input_event=trigger_mat)
                    if break_flag:
                        break

                # -------Ending Routine "QA_rec"-------
                for thisComponent in QA_recComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)

                q_a_rec_file = folder_path + 'rec_cali_de_pre.wav'
                q_a_rec_file = folder_path + 'rec_QA_run_' + str(run.thisN).zfill(2) + '_block_'+ str(QA_block.thisN).zfill(3) + '_trial_' + str(QA_trial.thisN).zfill(3)  + '.wav'
                write(q_a_rec_file, fs, QA_rec['recording'].file)  # Save as WAV file
                print('Recording is saved!' + q_a_rec_file)
                QA_trial = data_writer(QA_trial, QA_rec, 'QA_rec',
                    ['text', 'beep_hint', 'question', 'beep_start', 'beep_end', 'break'])

                print('Log: QA rec finish: Run ' + str(run.thisN) + ' Block ' + str(QA_block.thisN) + 'Trial ' + str(QA_trial.thisN))
                breakpoint_logger(comp='QA_rec', value=0, run=run.thisN, block=QA_block.thisN, trial=QA_trial.thisN)
                jxu_logging.info(
                    'Run-%s, Block-%s, Trial-%s, Q_cnt-%s, Q_sen-%s, Q_cen-%s' %(
                        run.thisN, QA_block.thisN, QA_trial.thisN, qa_question_cnt, QA_rec['question'].sen_text, QA_rec['question'].cen_text ))
                thisExp.addData('filename', q_a_rec_file)
                thisExp.nextEntry()
                trigger_sending(event_dict['QA_trial'][1], default_sleep=True) # Sending trigger 43 (QA_trial End)

        trigger_sending(event_dict['Block'][1], default_sleep=True) # Sending trigger 7 (Block End)
        # completed 3 repeats of 'QA_trial'
        # ---------------------------------------------------
        # --------------------- Pause -----------------------
        # ---------------------------------------------------
        if Pause_flag:
            # ------Prepare to start Routine "Pause"-------
            # update component parameters for each repeat
            Pause['key_resp'].keys = []
            Pause['key_resp'].rt = []
            Pause['cont'].setText('Press [space] key to continue.')
            # keep track of which components have finished
            win, Pause, PauseComponents, t, frameN, continueRoutine = pre_run_comp(win, Pause)
            # -------Run Routine "Pause"-------
            trigger_sending(event_dict['Pause'][0], default_sleep=True) # Sending trigger 60 (Pause Start)
            while continueRoutine:
                # get current time
                frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                    Pause["clock"], win, frameN)
                # update/draw components on each frame
                # *Pause['audio']* updates
                win, Pause['audio'], trigger = run_comp(
                    win, Pause['audio'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                    start_time=Pause['time']['audio'][0], duration=Pause['time']['audio'][1])
                # *Cali_de_pre_intro['key_resp']* updates
                waitOnFlip=False
                win, Pause['key_resp'], continueRoutine, endExpNow, trigger = run_comp(
                    win, Pause['key_resp'], 'key_resp', frameN, t, tThisFlip, tThisFlipGlobal,
                    start_time=Pause['time']['key_resp'][0], duration=Pause['time']['key_resp'][1],
                    waitOnFlip=waitOnFlip)
                # *Cali_de_pre_intro['cont']* updates
                win, Pause['cont'], trigger = run_comp(
                    win, Pause['cont'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                    start_time=Pause['time']['cont'][0], duration=Pause['time']['cont'][1],
                    repeat_per_frame=True, repeat_content='Please have some rest and press space key to continue')

                win, continueRoutine, break_flag = continue_justification(
                    win, endExpNow, defaultKeyboard, continueRoutine, PauseComponents)
                if break_flag:
                    break
            # -------Ending Routine "Pause"-------
            for thisComponent in PauseComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            QA_block.addData('Pause_cont.started', Pause['cont'].tStartRefresh)
            QA_block.addData('Pause_cont.stopped', Pause['cont'].tStopRefresh)

            QA_block = data_writer(QA_block, Pause, 'Pause', ['cont'])
            # the Routine "Pause" was not non-slip safe, so reset the non-slip timer
            trigger_sending(event_dict['Pause'][1], default_sleep=True) # Sending trigger 60 (Pause End)
            routineTimer.reset()
        thisExp.nextEntry()

    # completed 2 repeats of 'QA_block'


    # ---------------------------------------------------
    # --------------------- fade out ---------------------
    # ---------------------------------------------------
    if fade_out_flag:
        print('Log: fade out start: Run ' + str(run.thisN))
        trigger_sending(event_dict['Stable_stim'][1], default_sleep=True) # Sending trigger 29 (Stable_stim End)
        breakpoint_logger(comp='fade_out', value=1, run=run.thisN, block=None, trial=None)
        fade_in['text'].setText('Please remain seated until further instruction via the earphone.')
        # Instantiate the object for signal generator
        try:
            if isinstance(fg, SG()):
                pass
        except:
            fg = SG()

        if run.thisN == stim_run[-1]:
            intensity_change_flag = 'd'
        else:
            intensity_change_flag = 'keep'
        # if np.abs(fg.get_amp()- max) > 0.01 and run.thisN in stim_run:
        #     intensity_change_flag = 'i'


        # ------Prepare to start Routine "fade_out"-------
        # keep track of which components have finished



        # To be able to enter to loop of maintain/update intensity
        tmp_intensity = None
        if not input_intensity < max_intensity:
            tmp_intensity = input_intensity
            input_intensity = max_intensity - 0.05

        if intensity_change_flag == 'd':
            trigger_sending(event_dict['Fade_out'][0], default_sleep=True) # Sending trigger 26 (Fade_in Start)
        fade_itr_cnt = 0
        while input_intensity > min_intensity and fade_itr_cnt < n_step_fade_stim:
            if tmp_intensity != None:
                input_intensity = tmp_intensity
                tmp_intensity = None
                print('initial ' + str(input_intensity))

            win, fade_in, fade_inComponents, t, frameN, continueRoutine = pre_run_comp(win, fade_in)
            trigger_mat = np.zeros((len(fade_inComponents) - 1, 2))
            comp_list = np.asarray([*fade_in['time'].keys()])

            routineTimer.reset()
            routineTimer.add(fade_in['time']['auto_stim'][1])

            # -------Run Routine "fade_in"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                    fade_in["clock"], win, frameN)
                # *fade_in["text"]* updates
                win, fade_in['text'], trigger_mat[0] = run_comp(
                    win, fade_in['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                    start_time=fade_in['time']['text'][0], duration=fade_in['time']['text'][1])

                # Only maintain or update the stim intensity during stim run
                if stim_freq != 0 and run.thisN == stim_run[0]:
                    win, fade_in['auto_stim'], output_intensity, stim_continue, continueRoutine, endExpNow, intensity_change_flag, trigger_mat[1] = run_comp(
                        win, fade_in['auto_stim'], 'auto_stim', frameN, t, tThisFlip, tThisFlipGlobal,
                        start_time=fade_in['time']['auto_stim'][0], duration=fade_in['time']['auto_stim'][1],
                        stim_current_intensity=input_intensity, stim_intensity_limit=[min_intensity, max_intensity],
                        stim_step_intensity=step_intensity, stim_obj=fg, intensity_change_flag=intensity_change_flag,
                        stim=True)
                    input_intensity = output_intensity

                break_flag = False
                win, continueRoutine, break_flag = continue_justification(
                    win, endExpNow, defaultKeyboard, continueRoutine, fade_inComponents)

                if trigger_mat.sum(axis=0)[0]:
                    pass
                if break_flag:
                    break
            if stim_freq != 0 and run.thisN == stim_run[-1]:
                intensity_change_flag = 'd'
            else:
                intensity_change_flag = 'keep'
            fade_itr_cnt += 1

        # -------Ending Routine "fade_in"-------
        for thisComponent in fade_inComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)

        thisExp = data_writer(thisExp, fade_in, 'fade_out', ['text'])

        if intensity_change_flag == 'd':
            trigger_sending(event_dict['Fade_out'][1], default_sleep=True) # Sending trigger 27 (Fade_in End)

        # Fade in trigger: If sham stim, send trigger 22, otherwise 24
        if run.thisN == stim_run[-1]:
            if stim_freq == 0:
                trigger_sending(event_dict['Sham'][1], default_sleep=True) # Sending trigger 23 (Sham stim Start)
                win.flip()
            else:
                trigger_sending(event_dict['Stim'][1], default_sleep=True) # Sending trigger 21 (Real stim Start)
                fg.off()

        print('Log: fade out end: Run ' + str(run.thisN))
        breakpoint_logger(comp='fade_out', value=0, run=run.thisN, block=None, trial=None)
        routineTimer.reset()


    trigger_sending(event_dict['Run'][1], default_sleep=True) # Sending trigger 5 (Run End)
    thisExp.nextEntry()

# completed 3 repeats of 'run'

trigger_sending(event_dict['Post_run'][0], default_sleep=True) # Sending trigger 8 (Post_run Start)
# ---------------------------------------------------
# -------------- Cali_de_post_intro -----------------
# ---------------------------------------------------
if Cali_de_post_intro_flag:
    print('Log: cali_post_intro start')
    breakpoint_logger(comp='Cali_de_post_rec', value=1, run=None, block=None, trial=None)
    # ------Prepare to start Routine "Cali_de_post_intro"-------
    # update component parameters for each repeat
    Cali_de_post_intro['key_resp'].keys = []
    Cali_de_post_intro['key_resp'].rt = []
    # keep track of which components have finished
    win, Cali_de_post_intro, Cali_de_post_introComponents, t, frameN, continueRoutine = pre_run_comp(win, Cali_de_post_intro)
    trigger_mat = np.zeros((len(Cali_de_post_introComponents) - 1, 2))
    comp_list = np.asarray([*Cali_de_post_intro['time'].keys()])
    trigger_sending(10)  # Sending trigger 0 (Pre-Run Start)
    # -------Run Routine "Cali_de_post_intro"-------
    while continueRoutine:
        # get current time
        frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
            Cali_de_post_intro["clock"], win, frameN)
        # update/draw components on each frame
        # *Cali_de_post_intro["title"]* updates
        win, Cali_de_post_intro['title'], trigger_mat[0] = run_comp(
            win, Cali_de_post_intro['title'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Cali_de_post_intro['time']['title'][0], duration=Cali_de_post_intro['time']['title'][1])
        # *Cali_de_post_intro["title"]* updates
        win, Cali_de_post_intro['text'], trigger_mat[1] = run_comp(
            win, Cali_de_post_intro['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Cali_de_post_intro['time']['text'][0], duration=Cali_de_post_intro['time']['text'][1])
        # *Cali_de_post_intro["audio"]* updates
        win, Cali_de_post_intro['audio'], trigger_mat[2] = run_comp(
            win, Cali_de_post_intro['audio'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Cali_de_post_intro['time']['audio'][0], duration=Cali_de_post_intro['time']['audio'][1])

        # *Cali_de_post_intro['key_resp']* updates
        waitOnFlip=False
        win, Cali_de_post_intro['key_resp'], continueRoutine, endExpNow, trigger_mat[3] = run_comp(
            win, Cali_de_post_intro['key_resp'], 'key_resp', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Cali_de_post_intro['time']['key_resp'][0], duration=Cali_de_post_intro['time']['key_resp'][1],
            waitOnFlip=waitOnFlip)
        # *Cali_de_post_intro['cont']* updates
        win, Cali_de_post_intro['cont'], trigger_mat[4] = run_comp(
            win, Cali_de_post_intro['cont'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Cali_de_post_intro['time']['cont'][0], duration=Cali_de_post_intro['time']['cont'][1],
            repeat_per_frame=True, repeat_content=continue_str)

        win, continueRoutine, break_flag = continue_justification(
            win, endExpNow, defaultKeyboard, continueRoutine, Cali_de_post_introComponents)

        if trigger_mat.sum(axis=0)[0]:
            pass # trigger_encoding_sending('Calibration', input_run=3, input_block=0, intro_rec=0, input_event=trigger_mat)
        if break_flag:
            break
    trigger_sending(11)  # Sending trigger 0 (Pre-Run Start)
    time.sleep(0.003)
    # -------Ending Routine "Cali_de_post_intro"-------
    for thisComponent in Cali_de_post_introComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)

    thisExp = data_writer(thisExp, Cali_de_post_intro, 'Cali_de_post_intro', ['title', 'text', 'audio', 'cont'])
    # the Routine "Cali_de_post_intro" was not non-slip safe, so reset the non-slip timer
    print('Log: cali_post_intro finish')
    breakpoint_logger(comp='Cali_de_post_rec', value=0, run=None, block=None, trial=None)
    routineTimer.reset()


# ---------------------------------------------------
# --------------- Cali_de_post_rec -------------------
# ---------------------------------------------------
if Cali_de_post_rec_flag:
    # ---------------------------------------------------------------------------
    # ------------------------ Start Calibration Trial --------------------------
    # ---------------------------------------------------------------------------
    # set up handler to look after randomisation of conditions etc

    cali_post_trial = data.TrialHandler(nReps=n_cali_trial, method='random',
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='post_trial')
    thisExp.addLoop(cali_post_trial)  # add the loop to the experiment
    thisCali_post_trial = cali_post_trial.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisCali_post_trial.rgb)
    if thisCali_post_trial != None:
        for paramName in thisCali_post_trial:
            exec('{} = thisCali_post_trial[paramName]'.format(paramName))

    for thisCali_post_trial in cali_post_trial:

        if break_cali_post_trial != None and cali_post_trial.thisN < break_cali_post_trial:
            cali_question_cnt += 1
            continue

        currentLoop = cali_post_trial
        # abbreviate parameter names if possible (e.g. rgb = thisCali_post_trial.rgb)
        if thisCali_post_trial != None:
            for paramName in thisCali_post_trial:
                exec('{} = thisCali_post_trial[paramName]'.format(paramName))

        print('Log: cali_post_rec start: Trial ' + str(cali_post_trial.thisN))
        breakpoint_logger(comp='Cali_de_post_rec', value=1, run=None, block=None, trial=cali_post_trial.thisN)

        if external_question_flag:
            cali_question_cnt += 1
            Cali_de_post_rec['question_text'].setText(cali_sen_text[cali_question_cnt])
        else:
            Cali_de_post_rec['question_text'].setText('Text ' + str(cali_post_trial.thisN))

        # keep track of which components have finished
        win, Cali_de_post_rec, Cali_de_post_recComponents, t, frameN, continueRoutine = pre_run_comp(win, Cali_de_post_rec)
        trigger_mat = np.zeros((len(Cali_de_post_recComponents) - 1, 2))
        comp_list = np.asarray([*Cali_de_post_rec['time'].keys()])

        # -------Run Routine "Cali_de_post_rec"-------
        
        trigger_sending(event_dict['Cali_trial'][0], default_sleep=True) # Sending trigger 12 (Cali_trial Start)
        routineTimer.reset()
        routineTimer.add(cali_total_time)
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                Cali_de_post_rec["clock"], win, frameN)

            # *Cali_de_post_rec["text"]* updates
            win, Cali_de_post_rec['text'], trigger_mat[0] = run_comp(
                win, Cali_de_post_rec['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Cali_de_post_rec['time']['text'][0], duration=Cali_de_post_rec['time']['text'][1])
            # *Cali_de_post_rec["beep_hint"]* updates
            win, Cali_de_post_rec['beep_hint'], trigger_mat[1] = run_comp(
                win, Cali_de_post_rec['beep_hint'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Cali_de_post_rec['time']['beep_hint'][0], duration=Cali_de_post_rec['time']['beep_hint'][1])

            # *Cali_de_post_rec["question_text"]* updates
            win, Cali_de_post_rec['question_text'], trigger_mat[2] = run_comp(
                win, Cali_de_post_rec['question_text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Cali_de_post_rec['time']['question_text'][0], duration=Cali_de_post_rec['time']['question_text'][1])

            # *Cali_de_post_rec["beep_start"]* updates
            win, Cali_de_post_rec['beep_start'], trigger_mat[3] = run_comp(
                win, Cali_de_post_rec['beep_start'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Cali_de_post_rec['time']['beep_start'][0], duration=Cali_de_post_rec['time']['beep_start'][1])
            # *Cali_de_post_rec["recording"]* updates
            win, Cali_de_post_rec['recording'], trigger_mat[4] = run_comp(
                win, Cali_de_post_rec['recording'], 'recording', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Cali_de_post_rec['time']['recording'][0], duration=Cali_de_post_rec['time']['recording'][1])
            # *Cali_de_post_rec["beep_end"]* updates
            win, Cali_de_post_rec['beep_end'], trigger_mat[5] = run_comp(
                win, Cali_de_post_rec['beep_end'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Cali_de_post_rec['time']['beep_end'][0], duration=Cali_de_post_rec['time']['beep_end'][1])

            win, Cali_de_post_rec['break'], trigger_mat[6] = run_comp(
                win, Cali_de_post_rec['break'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Cali_de_post_rec['time']['break'][0], duration=Cali_de_post_rec['time']['break'][1])

            win, continueRoutine, break_flag = continue_justification(
                win, endExpNow, defaultKeyboard, continueRoutine, Cali_de_post_recComponents)

            if trigger_mat.sum(axis=0)[0]:
                trigger_encoding_sending('Calibration', input_event=trigger_mat)
            if break_flag:
                break
        # trigger_encoding_sending('Calibration', input_run=0, input_block=0, intro_rec=1, input_event=6)
        # -------Ending Routine "Cali_de_post_rec"-------
        for thisComponent in Cali_de_post_recComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)

        thisExp = data_writer(thisExp, Cali_de_post_rec, 'Cali_de_post_rec', ['text', 'beep_hint', 'question_text', 'beep_start', 'beep_end', 'break'])

        # cali_post_rec_file = folder_path + 'rec_cali_de_post_' + + ' .wav'
        cali_post_rec_file = folder_path + 'rec_cali_de_post_' + 'trial_' + str(cali_post_trial.thisN).zfill(3)  + '.wav'

        write(cali_post_rec_file, fs, Cali_de_post_rec['recording'].file)  # Save as WAV file
        print('Recording is saved!' + cali_post_rec_file)
        # Add the detected time into the PsychoPy data file:
        thisExp.addData('filename', cali_post_rec_file)

        thisExp.nextEntry()
        # the Routine "Cali_de_post_rec" was not non-slip safe, so reset the non-slip timer
        print('Log: cali_post_rec finish: Trial ' + str(cali_post_trial.thisN))
        breakpoint_logger(comp='Cali_de_post_rec', value=0, run=None, block=None, trial=cali_post_trial.thisN)
        routineTimer.reset()
        trigger_sending(event_dict['Cali_trial'][1], default_sleep=True) # Sending trigger 13 (Cali_trial End)


    # ---------------------------------------------------
    # --------------------- Pause -----------------------
    # ---------------------------------------------------
    if Pause_flag:
        # ------Prepare to start Routine "Pause"-------
        # update component parameters for each repeat
        Pause['key_resp'].keys = []
        Pause['key_resp'].rt = []
        Pause['cont'].setText('Press [space] key to continue.')
        # keep track of which components have finished
        win, Pause, PauseComponents, t, frameN, continueRoutine = pre_run_comp(win, Pause)
        # -------Run Routine "Pause"-------
        trigger_sending(event_dict['Pause'][0], default_sleep=True) # Sending trigger 60 (Pause Start)
        while continueRoutine:
            # get current time
            frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                Pause["clock"], win, frameN)
            # update/draw components on each frame

            # *Cali_de_pre_intro['key_resp']* updates
            waitOnFlip=False
            win, Pause['key_resp'], continueRoutine, endExpNow, trigger = run_comp(
                win, Pause['key_resp'], 'key_resp', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Pause['time']['key_resp'][0], duration=Pause['time']['key_resp'][1],
                waitOnFlip=waitOnFlip)
            # *Cali_de_pre_intro['cont']* updates
            win, Pause['cont'], trigger = run_comp(
                win, Pause['cont'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Pause['time']['cont'][0], duration=Pause['time']['cont'][1],
                repeat_per_frame=True, repeat_content='Please have some rest and press space key to continue')

            win, continueRoutine, break_flag = continue_justification(
                win, endExpNow, defaultKeyboard, continueRoutine, PauseComponents)
            if break_flag:
                break
        # -------Ending Routine "Pause"-------
        for thisComponent in PauseComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Pause_cont.started', Pause['cont'].tStartRefresh)
        thisExp.addData('Pause_cont.stopped', Pause['cont'].tStopRefresh)

        thisExp = data_writer(thisExp, Pause, 'Pause', ['cont'])
        trigger_sending(event_dict['Pause'][1], default_sleep=True) # Sending trigger 61 (Pause End)

        # the Routine "Pause" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()

trigger_sending(event_dict['Post_run'][1], default_sleep=True) # Sending trigger 9 (Post_run End)







# ---------------------------------------------------
# ---------------- Artifact_intro -------------------
# ---------------------------------------------------

if Artifact_intro_flag:

    print('Log: artifact_intro start')
    breakpoint_logger(comp='Artifact_intro', value=1, run=None, block=None, trial=None)
    # ------Prepare to start Routine "Cali_de_pre_intro"-------
    # update component parameters for each repeat
    Artifact_intro['key_resp'].keys = []
    Artifact_intro['key_resp'].rt = []
    Artifact_intro['audio'].setSound(audio_root+'artifact/intro.wav',secs=-1, hamming=True)

    # keep track of which components have finished
    win, Artifact_intro, Artifact_introComponents, t, frameN, continueRoutine = pre_run_comp(win, Artifact_intro)
    trigger_mat = np.zeros((len(Artifact_introComponents) - 1, 2))
    comp_list = np.asarray([*Artifact_intro['time'].keys()])

    # -------Run Routine "Artifact_intro"-------

    trigger_sending(event_dict['Arti_intro'][0], default_sleep=True) # Sending trigger 70 (Artifact_intro Start)
    while continueRoutine:
        # get current time
        frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
            Artifact_intro["clock"], win, frameN)
        # update/draw components on each frame

        # *Artifact_intro["title"]* updates
        win, Artifact_intro['title'], trigger_mat[0] = run_comp(
            win, Artifact_intro['title'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Artifact_intro['time']['title'][0], duration=Artifact_intro['time']['title'][1])
        # *Artifact_intro["title"]* updates
        win, Artifact_intro['text'], trigger_mat[1] = run_comp(
            win, Artifact_intro['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Artifact_intro['time']['text'][0], duration=Artifact_intro['time']['text'][1])
        # *Artifact_intro["audio"]* updates
        win, Artifact_intro['audio'], trigger_mat[2] = run_comp(
            win, Artifact_intro['audio'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Artifact_intro['time']['audio'][0], duration=Artifact_intro['time']['audio'][1])

        # *Artifact_intro['key_resp']* updates
        waitOnFlip=False
        win, Artifact_intro['key_resp'], continueRoutine, endExpNow, trigger_mat[3] = run_comp(
            win, Artifact_intro['key_resp'], 'key_resp', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Artifact_intro['time']['key_resp'][0], duration=Artifact_intro['time']['key_resp'][1],
            waitOnFlip=waitOnFlip)
        # *Artifact_intro['cont']* updates
        win, Artifact_intro['cont'], trigger_mat[4] = run_comp(
            win, Artifact_intro['cont'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=Artifact_intro['time']['cont'][0], duration=Artifact_intro['time']['cont'][1],
            repeat_per_frame=True, repeat_content=continue_str)

        win, continueRoutine, break_flag = continue_justification(
            win, endExpNow, defaultKeyboard, continueRoutine, Artifact_introComponents)
        if trigger_mat.sum(axis=0)[0]:
            pass # trigger_encoding_sending('Calibration', input_run=0, input_block=0, intro_rec=0, input_event=trigger_mat)
        if break_flag:
            break

    # -------Ending Routine "Artifact_intro"-------
    for thisComponent in Artifact_introComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)

    thisExp = data_writer(thisExp, Artifact_intro, 'Artifact_intro', ['title', 'text', 'audio', 'cont'])

    trigger_sending(event_dict['Arti_intro'][1], default_sleep=True) # Sending trigger 71 (Artifact_intro End)
    print('Log: artifact_intro finish')
    breakpoint_logger(comp='Artifact_intro', value=0, run=None, block=None, trial=None)
    routineTimer.reset()



# ---------------------------------------------------
# ----------------- Artifact_rec --------------------
# ---------------------------------------------------


if Artifact_rec_flag:
    artifact_action_cnt = -1
    # ---------------------------------------------------------------------------
    # ------------------------ Start Artifact_rec Trial --------------------------
    # ---------------------------------------------------------------------------
    # set up handler to look after randomisation of conditions etc

    artifact_trial = data.TrialHandler(nReps=n_arti_trial, method='random',
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='artifact_trial')
    thisExp.addLoop(artifact_trial)  # add the loop to the experiment
    thisArtifact_trial = artifact_trial.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisCali_pre_trial.rgb)
    if thisArtifact_trial != None:
        for paramName in thisArtifact_trial:
            exec('{} = thisArtifact_trial[paramName]'.format(paramName))

    for thisArtifact_trial in artifact_trial:

        currentLoop = artifact_trial
        # abbreviate parameter names if possible (e.g. rgb = thisArtifact_trial.rgb)
        if thisArtifact_trial != None:
            for paramName in thisArtifact_trial:
                exec('{} = thisArtifact_trial[paramName]'.format(paramName))

        # ------Prepare to start Routine "Artifact_rec"-------
        print('Log: artifact_rec start: Trial ' + str(artifact_trial.thisN))
        breakpoint_logger(comp='Artifact_rec', value=1, run=None, block=None, trial=artifact_trial.thisN)

        artifact_action_cnt += 1
        Artifact_rec['action'].setSound(action_path_list[artifact_action_cnt], secs=-1, hamming=True)

        # keep track of which components have finished
        win, Artifact_rec, Artifact_recComponents, t, frameN, continueRoutine = pre_run_comp(win, Artifact_rec)
        trigger_mat = np.zeros((len(Artifact_recComponents) - 1, 2))
        comp_list = np.asarray([*Artifact_rec['time'].keys()])

        # -------Run Routine "Artifact_rec"-------
        trigger_sending(event_dict['Arti_trial'][0], default_sleep=True) # Sending trigger 72 (Artifact_trial Start)
        routineTimer.reset()
        routineTimer.add(artifact_total_time)
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
                Artifact_rec["clock"], win, frameN)

            # *Artifact_rec["text"]* updates
            win, Artifact_rec['text'], trigger_mat[0] = run_comp(
                win, Artifact_rec['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Artifact_rec['time']['text'][0], duration=Artifact_rec['time']['text'][1])
            # *Artifact_rec["beep_hint"]* updates
            win, Artifact_rec['beep_hint'], trigger_mat[1] = run_comp(
                win, Artifact_rec['beep_hint'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Artifact_rec['time']['beep_hint'][0], duration=Artifact_rec['time']['beep_hint'][1])
            # *Artifact_rec["action"]* updates
            win, Artifact_rec['action'], trigger_mat[2] = run_comp(
                win, Artifact_rec['action'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Artifact_rec['time']['action'][0], duration=Artifact_rec['time']['action'][1])

            # *Artifact_rec["beep_start"]* updates
            win, Artifact_rec['beep_start'], trigger_mat[3] = run_comp(
                win, Artifact_rec['beep_start'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Artifact_rec['time']['beep_start'][0], duration=Artifact_rec['time']['beep_start'][1])
            # *Artifact_rec["recording"]* updates
            win, Artifact_rec['recording'], trigger_mat[4] = run_comp(
                win, Artifact_rec['recording'], 'recording', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Artifact_rec['time']['recording'][0], duration=Artifact_rec['time']['recording'][1])
            # *Artifact_rec["beep_end"]* updates
            win, Artifact_rec['beep_end'], trigger_mat[5] = run_comp(
                win, Artifact_rec['beep_end'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Artifact_rec['time']['beep_end'][0], duration=Artifact_rec['time']['beep_end'][1])

            win, Artifact_rec['break'], trigger_mat[6] = run_comp(
                win, Artifact_rec['break'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
                start_time=Artifact_rec['time']['break'][0], duration=Artifact_rec['time']['break'][1])

            win, continueRoutine, break_flag = continue_justification(
                win, endExpNow, defaultKeyboard, continueRoutine, Artifact_recComponents)

            if trigger_mat.sum(axis=0)[0]:
                trigger_encoding_sending('Artifact', input_event=trigger_mat)
            if break_flag:
                break
        # -------Ending Routine "Artifact_rec"-------
        for thisComponent in Artifact_recComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)

        thisExp = data_writer(thisExp, Artifact_rec, 'Artifact_rec', ['text', 'beep_hint', 'action', 'beep_start', 'recording', 'beep_end', 'break'])
        thisExp.nextEntry()

        trigger_sending(event_dict['Arti_trial'][1], default_sleep=True) # Sending trigger 73 (Artifact_trial End)
        print('Log: artifact_rec finish: Trial' + str(artifact_trial.thisN))
        breakpoint_logger(comp='Artifact_rec', value=0, run=None, block=None, trial=artifact_trial.thisN)
        # the Routine "Artifact_rec" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()






















# ---------------------------------------------------
# ---------------------- End ------------------------
# ---------------------------------------------------
if end_flag:

    # ------Prepare to start Routine "the_end"-------
    # update component parameters for each repeat
    # keep track of which components have finished
    win, the_end, the_endComponents, t, frameN, continueRoutine = pre_run_comp(win, the_end)
    # -------Run Routine "the_end"-------
    while continueRoutine:
        # get current time
        frameN, t, tThisFlip, tThisFlipGlobal, win = time_update(
            the_end["clock"], win, frameN)
        # update/draw components on each frame
        # *the_end['audio']* updates
        win, the_end['audio'], trigger = run_comp(
            win, the_end['audio'], 'audio', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=the_end['time']['audio'][0], duration=the_end['time']['audio'][1])

        # *the_end['cont']* updates
        win, the_end['text'], trigger = run_comp(
            win, the_end['text'], 'text', frameN, t, tThisFlip, tThisFlipGlobal,
            start_time=the_end['time']['text'][0], duration=the_end['time']['text'][1],
            repeat_per_frame=True, repeat_content=the_end['text'].text)

        win, continueRoutine, break_flag = continue_justification(
            win, endExpNow, defaultKeyboard, continueRoutine, the_endComponents)
        if break_flag:
            break
    # -------Ending Routine "Pause"-------
    for thisComponent in the_endComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('the_end_text.started', the_end['text'].tStartRefresh)
    thisExp.addData('the_end_text.stopped', the_end['text'].tStopRefresh)

    thisExp = data_writer(thisExp, the_end, 'the_end', ['text'])

    # the Routine "Pause" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()

trigger_sending(event_dict['End'], default_sleep=True)  # trigger 255 represents experiment ends









# Flip one final time so any remaining win.callOnFlip()
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()

