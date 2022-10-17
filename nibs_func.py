
from __future__ import absolute_import, division


from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']
from psychopy import locale_setup
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
from pdb import set_trace
# What signaler class to use? Here just the demo signaler:
from psychopy.voicekey.demo_vks import DemoVoiceKeySignal as Signaler
import sounddevice as sd
from scipy.io.wavfile import write
import parallel
import numbers
import pandas as pd
import time
from datetime import datetime





def exp_init(Name='nibs_stage_1', fade_flag=True, pause_flag=True):
    # -----------------------------------------------------------------------------------
    # -------------------- Experiment & Device initialization ---------------------------
    # -----------------------------------------------------------------------------------
    # Store info about the experiment session
    expName = Name  # from the Builder filename that created this script
    
    Info = {'participant': '00',
            'session': '00',
            'First language': '',
            'German level': ['A1', 'A2', 'B1', 'B2'],
            'Full_screen': False,
            'Instruction': True,
            'Intro': True,
            'Artifact': True,
            'Cali_pre': True,
            'Fade_in_out': fade_flag,
            'Artifact_within': True,
            'Resting_State': True,
            'QA': True,
            'Pause': pause_flag,
            'Cali_post': True,
            'End': True,
            'Breakpoint_Cali_pre_trial': 'No',
            'Breakpoint_Run': ['No', 0, 1, 2, 3],
            'Breakpoint_RS_block': ['No', 0, 1],
            'Breakpoint_QA_block': ['No', 0, 1],
            'Breakpoint_QA_trial': 'No',
            'Breakpoint_Cali_post_trial': 'No'
            }
    dlg = gui.DlgFromDict(dictionary=Info, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    Info['date'] = data.getDateStr()  # add a simple timestamp
    Info['expName'] = expName
    Info['psychopyVersion'] = '3.2.3'

    if Info['Full_screen']:
        win_size = [3840, 2160]
    else:
        win_size = [960, 540]

    # Setup the Window
    win = visual.Window(
        size=win_size, fullscr=False, screen=0, 
        winType='pyglet', allowGUI=True, allowStencil=False,
        monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
        blendMode='avg', useFBO=True)
    # store frame rate of monitor if we can measure it
    Info['frameRate'] = win.getActualFrameRate()
    if Info['frameRate'] != None:
        frameDur = 1.0 / round(Info['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess

    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard()

    return Info, win, frameDur, defaultKeyboard 


def create_folder(folder_path):
    # copy from "from jxu.basiccmd.cmd import create_folder" 
    if os.path.exists(folder_path):
        return print("Traget folder is already created! Path: \n" + folder_path)
    else:
        try:
            os.mkdir(folder_path)
        except:
            path = os.path.normpath(folder_path)
            path_seg = path.split(os.sep)
            if len(path_seg) == 0:
                raise ValueError("Empty path of target folder!")

            for subfolder_path in path_seg:
                if not os.path.exists(subfolder_path):
                    os.mkdir(subfolder_path)
                os.chdir(subfolder_path)
            os.chdir(os.path.dirname(os.path.abspath(os.getcwd())))

        return print("Target folder is successfully created! Path: \n" + folder_path)

def path_init(expInfo):
    # -----------------------------------------------------------------------------------
    # ----------------------- Folder & File initialization ------------------------------
    # -----------------------------------------------------------------------------------
    _thisDir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(_thisDir)

    folder_path = './data/Subject_%s/Session_%s/Exp_data/%s/' %(
        expInfo['participant'], expInfo['session'], expInfo['date'])
    filename = _thisDir + os.sep + 'data/Subject_%s/Session_%s/%s_%s' %(
        expInfo['participant'], expInfo['session'], expInfo['expName'], expInfo['date'])
    create_folder(folder_path)

    return folder_path, filename


def component_init(routine, comp, comp_index):
    if comp['property'] == 'textstim':
        return visual.TextStim(name=routine + '_' + comp['comp_name'],
                               depth=-np.float(comp_index),
                               **comp['parameters'])
    elif comp['property'] == 'keyboard':
        return keyboard.Keyboard(**comp['parameters'])
    elif comp['property'] == 'audio':
        return sound.Sound(name=routine + '_' + comp['comp_name'],
                           **comp['parameters'])
    elif comp['property'] == 'recording':
        class rec: pass
        rec.status = None
        rec.name = routine + '_' + comp['comp_name']
        for k, v in comp['parameters'].items():
            setattr(rec, k, v)
        return rec
    elif comp['property'] == 'trigger':
        class trigger: pass
        for k, v in comp['parameters'].items():
            setattr(trigger, k, v)
        return trigger
    elif comp['property'] == 'textstim':
        pass
    elif comp['property'] == 'textstim':
        pass
    else:
        pass


def routine_init(routine_name, comp_list):
    routine = {}
    routine["clock"] = core.Clock()
    routine['property'] = {}
    for comp_ind, comp in enumerate(comp_list):
        routine[comp['comp_name']] = component_init(routine_name, comp, comp_ind)
        # routine['property'].update({comp['comp_name']: comp['property']})
    return routine


def trigger_generator(win, name):
    dict_textstim = {
        'property':'trigger',
        'comp_name': name,
        'parameters':{'status': None}
        }
    return dict_textstim

def textstim_generator(win, name, content='', pos=[0.5, 0.0], font_size=0.06, font_type='Arial', bold=False):
    dict_textstim = {
        'property':'textstim',
        'comp_name': name,
        'parameters':{'win': win,
                      'text':content,
                      'pos': pos,
                      'height': font_size,
                      'font': font_type,
                      'bold': bold}
        }
    return dict_textstim


def key_resp_generator(name):
    dict_key_resp = {
        'property':'keyboard',
        'comp_name': name,
        'parameters':{}
        }
    return dict_key_resp


def audio_generator(name, loc, secs=-1, vol=1.0, sr=44100, hamming=True):
    dict_audio = {
        'property':'audio',
        'comp_name': name,
        'parameters':{'value': loc,
                      'volume':vol,
                      'sampleRate': sr,
                      'hamming': hamming,
                      'secs': secs}
        }
    return dict_audio


def rec_generator(name, sps=44100, n_rec_chn=2, loc='./data/'):
    dict_rec = {
        'property':'recording',
        'comp_name': name,
        'parameters':{'fs': sps,
                      'file_path': loc,
                      'channels': n_rec_chn,
                      'samplerate': sps}
        }
    return dict_rec


# keep track of which components have finished
def pre_run_comp(win, obj):
    objComponents = [v for v in obj.values() if not isinstance(v, dict)]
    for thisComponent in objComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    obj['clock'].reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    continueRoutine = True

    return win, obj, objComponents, t, frameN, continueRoutine


def time_update(obj_clock, win, frame):
    current_time = obj_clock.getTime()
    current_routine_time = win.getFutureFlipTime(clock=obj_clock)
    current_global_time = win.getFutureFlipTime(clock=None)
    current_frame = frame + 1

    return current_frame, current_time, current_routine_time, current_global_time, win


def run_comp(win, obj, obj_property, current_frame, current_time, current_routine_time,
    current_global_time, start_time=0, duration=None, repeat_per_frame=False,
    repeat_content=None, frameTolerance=0.001, waitOnFlip=False, key_list=['space'],
    continueRoutine_flag=True, endExpNow_flag=False, fs=44100, rec_filepath=None, stim=False,
    stim_obj=None, stim_step_intensity=None, stim_current_intensity=None,
    stim_freq=None, stim_offset=None, stim_intensity_limit=None, stim_continue_flag=False,
    intensity_change_flag=None):

    trigger_flag = False
    trigger_val = None
    if obj.status == NOT_STARTED and current_routine_time >= start_time-frameTolerance:
        # keep track of start time/frame for later
        obj.frameNStart = current_frame  # exact frame index
        obj.tStart = current_time  # local t and not account for scr refresh
        obj.tStartRefresh = current_global_time  # on global time
        trigger_flag = True 
        trigger_val = 0
        if obj_property == 'text':
            win.timeOnFlip(obj, 'tStartRefresh')  # time at next scr refresh
            obj.setAutoDraw(True)
        elif obj_property == 'audio':
            obj.status = STARTED
            obj.play(when=win)  # sync with win flip
        elif obj_property == 'key_resp' or obj_property == 'key_resp_stim' :
            obj.status = STARTED
            win.callOnFlip(obj.clearEvents, eventType='keyboard')  # clear events on next screen flip
        elif obj_property == 'auto_stim':
            obj.status = STARTED
        elif obj_property == 'trigger':
            obj.status = STARTED
            trigger_sending(50)
        elif obj_property == 'recording':
            obj.status = STARTED
            try:
                print('Recording start!')
                try:
                    obj.file = sd.rec(int(duration * obj.fs), samplerate=obj.fs, channels=obj.channels)
                except:
                    print('here?')
            except:
                obj.file = None
                print('No predefined duration of recording!')

    if obj.status == STARTED:
        if repeat_per_frame or duration != None:
            if duration != None and current_global_time > obj.tStartRefresh + duration - frameTolerance:
                obj.tStop = current_time
                obj.frameNStop = current_frame
                win.timeOnFlip(obj, 'tStopRefresh')
                trigger_flag = True
                trigger_val = 1
                if obj_property == 'text':
                    obj.setAutoDraw(False)
                elif obj_property == 'audio':
                    obj.stop()
                elif obj_property == 'trigger':
                    trigger_sending(51)
                    obj.status = FINISHED
                elif obj_property == 'recording':
                    # print(sd.wait())
                    obj.status = FINISHED
            elif repeat_per_frame:
                if obj_property == 'text':
                    obj.setText(repeat_content, log=False)
                elif obj_property == 'audio':
                    pass
        if obj_property == 'auto_stim' and stim:
            stim_min_intensity, stim_full_intensity = stim_intensity_limit
            stim_continue_flag = stim_continue_flag
            if intensity_change_flag == None:
                raise ValueError("For auto_stim, a flag is needed which is either 'i' or 'd' or 'keep'")
            else:         
                if "i" == intensity_change_flag:
                    stim_current_intensity += stim_step_intensity
                elif "d" == intensity_change_flag:
                    stim_current_intensity -= stim_step_intensity
                else:
                    pass # print("Stim flag: " + intensity_change_flag)

                if intensity_change_flag != 'keep':
                    if stim_current_intensity > stim_full_intensity:
                        stim_current_intensity = stim_full_intensity
                    elif stim_current_intensity < stim_min_intensity:
                        stim_current_intensity = stim_min_intensity  # Minimum output of waveform generator
                    else:
                        stim_current_intensity = stim_current_intensity

                    if stim_current_intensity < 0.01:
                        stim_current_intensity = np.round(stim_current_intensity, 3)  # lowest limit of waveform generator
                    else:
                        stim_current_intensity = np.round(stim_current_intensity, 2)
                    stim_obj.amp(stim_current_intensity)

                    if stim_freq is not None:
                        stim_obj.frequency(stim_freq)
                    if stim_offset is not None:
                        stim_obj.offset(stim_offset)

                    continueRoutine_flag = True
                    intensity_change_flag = 'keep'

        if obj_property == 'key_resp' and not waitOnFlip:
            theseKeys = obj.getKeys(keyList=key_list, waitRelease=False)
            if len(theseKeys):
                theseKeys = theseKeys[0]  # at least one key was pressed
                # check for quit:
                if "escape" == theseKeys:
                    endExpNow_flag = True
                else:
                    endExpNow_flag = False
                continueRoutine_flag = False

        if obj_property == 'key_resp_stim' and not waitOnFlip and stim:
            theseKeys = obj.getKeys(keyList=key_list, waitRelease=False)
            stim_min_intensity, stim_full_intensity = stim_intensity_limit
            stim_continue_flag = stim_continue_flag
            if len(theseKeys):
                theseKeys = theseKeys[0]  # at least one key was pressed
                # check for quit:
                if "escape" == theseKeys:
                    endExpNow_flag = True
                elif "space" == theseKeys:
                    stim_continue_flag = True
                elif "i" == theseKeys:
                    stim_current_intensity += stim_step_intensity
                elif "d" == theseKeys:
                    stim_current_intensity -= stim_step_intensity
                else:
                    endExpNow_flag = False

                if stim_current_intensity > stim_full_intensity:
                    stim_current_intensity = stim_full_intensity
                elif stim_current_intensity < stim_min_intensity:
                    stim_current_intensity = stim_min_intensity  # Minimum output of waveform generator
                else:
                    stim_current_intensity = stim_current_intensity

                stim_current_intensity = np.round(stim_current_intensity, 2)
                stim_obj.amp(stim_current_intensity)

                print(stim_current_intensity)
                if stim_freq is not None:
                    stim_obj.frequency(stim_freq)
                if stim_offset is not None:
                    stim_obj.offset(stim_offset)

                continueRoutine_flag = False

    if obj_property == 'key_resp':
        return win, obj, continueRoutine_flag, endExpNow_flag, np.asarray([[trigger_flag, trigger_val]])
    elif obj_property == 'key_resp_stim':
        return win, obj, stim_current_intensity, stim_continue_flag, continueRoutine_flag, endExpNow_flag, np.asarray([[trigger_flag, trigger_val]])
    elif obj_property == 'auto_stim' :
        return win, obj, stim_current_intensity, stim_continue_flag, continueRoutine_flag, endExpNow_flag, intensity_change_flag, np.asarray([[trigger_flag, trigger_val]])
    else:
        return win, obj, np.asarray([[trigger_flag, trigger_val]])


def continue_justification(win, endExpNow_flag, defaultKeyboard, continueRoutine_flag, objComponents,
    break_flag=False):
    # check for quit (typically the Esc key)
    if endExpNow_flag or defaultKeyboard.getKeys(keyList=["escape"]):
        trigger_sending(1)  # Permanent reserved for exit
        core.quit()
    
    # check if all components have finished
    if not continueRoutine_flag:  # a component has requested a forced-end of Routine
        break_flag = True
    continueRoutine_flag = False  # will revert to True if at least one component still running
    for thisComponent in objComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine_flag = True
            break   # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine_flag:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

    return win, continueRoutine_flag, break_flag


def data_writer(target, obj, obj_str, list_kwgs, text_append=True):
    for kwgs in list_kwgs:
        if 'audio' in kwgs or 'beep' in kwgs or kwgs == 'question':
            obj[kwgs].stop() # ensure sound has stopped at end of routine
        if text_append:
            if kwgs == 'question':  # For audio stimuli
                target.addData(obj_str + '_' + kwgs + '.sen_text', obj[kwgs].sen_text)
                target.addData(obj_str + '_' + kwgs + '.cen_text', obj[kwgs].cen_text) 
            if kwgs == 'question_text' or kwgs == 'text':   # For visual stimuli
                target.addData(obj_str + '_' + kwgs + '.text', obj[kwgs].text)

        target.addData(obj_str + '_' + kwgs + '.started', obj[kwgs].tStartRefresh)
        target.addData(obj_str + '_' + kwgs + '.stopped', obj[kwgs].tStopRefresh)
    return target


def trigger_sending(data, port='/dev/parport0', default_sleep=None):
    try:
        value = np.uint8(data)
    except:
        return None  # print('Silent trigger!')

    try: 
        ext_port = parallel.Parallel(port=port)
        ext_port.setData(value)
        if default_sleep is not None:
            time.sleep(default_sleep) 
        return print('successfully send trigger: ' + str(value))
    except:
        return print('No device! Planned to send trigger: ' + str(value))


def trigger_encoding_sending(obj_name, input_event, port='/dev/parport0'):

    """
    0. Structure of Experiment
    =====================================================================================================
    | ------------- HEADER -------------- | ----------------------- DATA ------------------- | -FOOTER- |
    | START Exp. | Instruction | Practice | Pre-Run | Run 0 | Run 1 | ... | Run 3 | Post-Run | END Exp. |
    | ................................... | ^     ^ | ^   ^ | ^   ^ | ... | ^   ^ | ^      ^ | ........ |
    | ................................... | 0     1 | 4   5 | 4   5 | ... | 4   5 | 2      3 | ........ |
    =====================================================================================================


    1.1. Structure of Pre-/Post-Run (Calibration)  
    ===================================================================================
    |  -HEADER- | <--- 30 secs ---> | ........................... | ----- FOOTER ---- |
    |   Intro   |     C-Trial 0     | C-Trial 1 | ... | C-Tiral 9 | Pause (unlimited) |
    | < 1 min > | <------------------- 5 mins ------------------> | <----- n/a -----> |
    | ^       ^ | ^               ^ | ^       ^ | ... | ^       ^ | ^               ^ |
    | 10     11 | 12             13 | 12     13 | ... | 12     13 | 60             61 |
    ===================================================================================

   
    1.2. Structure of Pre-/Post-Run trial (Calibration)
    ================================================================================================
       | -------------------------------- Calibration Trail K --------------------------------- |
       | ---------------------------------------- DATA ---------------------------------------- |
       |   Q-Beep | Display sentence | Blank |  A-S-Beep |    A-Rec   |  A-S-Beep | -- Break -- |
       | <------------ 18 secs ------------> | < 1 sec > | < 8 secs > | < 1 sec > | < 2 secs  > |
    ^  |          | ^              ^ |       | ^         | ^        ^ |         ^ | ^         ^ |  ^
    12 |          | 14            15 |       | 16        | 18      19 |        17 | 62       63 | 13
    ================================================================================================


    2.1. Structure of Run

    Plan A: (not adopted)
    =============================================================================================================
    | ----------------------------------------------- Run K --------------------------------------------------- |
    | --------------------------------------------------------------------------------------------------------- |
    | <---- 1 min ----> | <-- 10mins --> | <-- ~12mins --> | <-- 10mins --> | <-- ~12mins --> | <--- 1 min ---> |
    |  START Stim/Sham  | Resting State  |     Block 0     | Resting State  |     Block 1     |  END Stim/Sham  |
    | <----- 1 min ---> | <---------------------------- ~45mins ----------------------------> | <--- 1 min ---> |
    | ^               ^ | ................................................................... | ^  ^          ^ |
    | 20 24       25 28 | ................................................................... | 29 26     27 21 |    
    | 22             28 | ................................................................... | 29           23 |    
    =============================================================================================================

    Plan B: (adopted)  --> Argument: Shorter duration (40 mins less than A) and RS is not major goal for now
    ============================================================================================
    | -------------------------------------- Run K ------------------------------------------- |
    | ---------------------------------------------------------------------------------------- |
    | <---- 1 min ----> | <-- 10mins --> | <-- ~12mins --> | <-- ~12mins --> | <--- 1 min ---> |
    |  START Stim/Sham  | Resting State  |     Block 0     |     Block 1     |  END Stim/Sham  |
    | <----- 1 min ---> | <------------------- ~35mins --------------------> | <--- 1 min ---> |
    | ^               ^ | ^            ^ | ^             ^ | ^             ^ | ^  ^          ^ |
    | 20 24       25 28 | 6            7 | 6             7 | 6             7 | 29 26     27 21 |    
    | 22             28 | .............. | ............... | ............... | 29           23 |    
    ============================================================================================

    2.1.1. Structure of Resting State
    ==========================================================================================
      | --------------------------------- Resting State ---------------------------------- |
      | -- HEADER -- | -------- DATA ---------- | -- HEADER --  | -------- DATA ---------- |
      | Intro (open) | Relax while opening eyes | Intro (close) | Relax while closing eyes |
      | < 30 secs >  | <------- 5 mins -------> |  < 30 secs >  | <------- 5 mins -------> | 
    ^ | ^         ^  | ^                      ^ |  ^         ^  | ^                      ^ | ^
    6 | 30       31  | 32                    33 |  30       31  | 34                    35 | 7
    ==========================================================================================


    2.1.2. Structure of Block  (For Block K, K!=0, no HEADER)  -> No need to repeat intro, since same tasks
    ====================================================================================
    | ------------------------------------ Block K --------------------------------- |
    |  -HEADER- |   | <-- 30 secs --> | ........................ | ----- FOOTER ---- |
    |   Intro   |   |     Trial 0     | Trial 1 | ... | Tiral 19 | Pause (unlimited) |
    | < 1 min > |   | <---------------- 10 mins ---------------> | <----- n/a -----> |
    | ^       ^ | ^ | ^             ^ | ........................ | ^               ^ | ^ 
    | 40     41 | 6 | 42           43 | ........................ | 60             61 | 7
    ====================================================================================


    2.1.3. Structure of Trial (Q&A)  --- Not implemented inside trial
    =============================================================================================================================
       | ------------------------------------------------- Trial K --------------------------------------------------------- |
       | -Beep- | --------------- Question Audio--------------- -------- | --------- Answer Recording -------- | ----------- |
       | Q-Beep | Q-Audio-1 | Word censored by 40Hz |  Q-Audio-2 | Blank |  A-S-Beep |    A-Rec   |  A-S-Beep  | -- Break -- |
       | <---------------------- 18 secs ------------------------------> | < 1 sec > | < 8 secs > | < 1 sec >  | < 2 secs  > |
    ^  |        | ^         | ^                   ^ |          ^ |       | ^         | ^        ^ |         ^  | ^         ^ |  ^
    42 |        | 44        | 50                 51 |         45 |       | 46        | 48      49 |        47  | 62       63 | 43
    =============================================================================================================================
    
    =============================
    |--- General Trigger (0) ---|
    =============================
    Pre-run   start/end: 0/1
    Post-Run  start/end: 2/3
    Run       start/end: 4/5
    Block     start/end: 6/7
 
    
    ========================================
    |--- Calibration Trial Trigger (10) ---|
    ========================================
    Intro             start/end: 0/1
    Trial             start/end: 2/3
    Display           start/end: 4/5
    Answer            start/end: 6/7
    Answer recording  start/end: 8/9

    =================================
    |--- Stim./Sham Trigger (20) ---|
    =================================
    Stim.         start/end: 0/1
    Sham          start/end: 2/3
    Fade in       start/end: 4/5
    Fade out      start/end: 6/7
    Stable stim.  start/end: 8/9


    ====================================
    |--- Resting State Trigger (30) ---|
    ====================================
    Intro                start/end: 0/1
    Opening eyes         start/end: 2/3
    Closing eyes         start/end: 4/5


    ==========================================  
    |--- Q&A Block& Trial Trigger (40) ---|
    ==========================================
    Intro             start/end: 0/1
    Trial             start/end: 2/3
    Q-audio           start/end: 4/5
    Answer            start/end: 6/7
    Answer recording  start/end: 8/9
    Censored word     start/end: 10/11


    ===================================
    |---  Break/Pause Trigger (60) ---|
    ===================================
    Pause  start/end: 0/1
    Break  start/end: 2/3 
    
    ==========================================  
    |--- Artifact collection Trigger (70) ---|
    ==========================================
    Intro             start/end: 0/1
    Trial             start/end: 2/3
    Q-audio           start/end: 4/5
    Answer            start/end: 6/7
    Answer recording  start/end: 8/9
    Censored word     start/end: 10/11

    """

    task = {'Calibration': 10, 'QA': 40, 'Artifact': 70}
    digit_task = task[obj_name]
    trigger_table = np.array([[None, None],
                              [None, None],
                              [4, 5],
                              [6, None],
                              [8, 9],
                              [None, 7],
                              [62, 63]])

    if isinstance(input_event, numbers.Number):
        digit_event = str(input_event)
        data = digit_task + input_event
        trigger_sending(data, port=port)
    else:
        for ind in np.where(input_event[:, 0] == 1)[0]:
            digit_event = input_event[ind, 1]
            if ind == trigger_table.shape[0] - 1:
                data = trigger_table[ind, int(digit_event)]
            else:
                try:
                    data = digit_task + trigger_table[ind, int(digit_event)]
                except:
                    data = None

            trigger_sending(data, port=port)


def extract_qa(input_all_df=None, label='practice', word_type='VERB',
    file_root='/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/', df_file='all_beep_df.pkl'):
    try:
        if input_all_df == None: 
            dataframe_path = file_root + df_file
            all_df = pd.read_pickle(dataframe_path)
    except:
        if not input_all_df.empty:
            all_df = input_all_df
        else:
            raise ValueError('No such file!')
    # No repeat questions for same subject
    if word_type != 'VERB':
        word_type_df = all_df[all_df['SENTENCE_INFO']['beep_word_type'] == word_type]
    
    file_loc_list = all_df['PATH']['file_root_syn'].values
    censor_start = all_df['SENTENCE_INFO']['beeped_word_timestamp_start'].values
    censor_dur = all_df['SENTENCE_INFO']['beeped_word_duration'].values
    sen_duration = all_df['SENTENCE_INFO']['sentence_duration'].values
    sen_text = all_df['SENTENCE_INFO']['sen_content'].values
    cen_text = all_df['SENTENCE_INFO']['beeped_word'].values

    return all_df, file_loc_list, censor_start, censor_dur, sen_duration, sen_text, cen_text



def extract_cali(input_all_df=None):

    if not input_all_df.empty:
        all_df = input_all_df
    else:
        raise ValueError('No such file!')
    # No repeat questions for same subject

    
    sen_text = all_df['sentence'].values
    sen_order = all_df['sentence'].values

    return sen_text, sen_order


def randomization_question():

    all_beep_df = pd.read_pickle('../audio_data/all_unshattered_beep_df.pkl')

    all_last_df = all_beep_df.loc[all_beep_df.SENTENCE_INFO.last_word_flag==True]
    all_mid_df = all_beep_df.loc[all_beep_df.SENTENCE_INFO.last_word_flag==False]

    last_pi = np.asarray(all_last_df.SENTENCE_INFO.permanent_index.values)
    mid_pi = np.asarray(all_mid_df.SENTENCE_INFO.permanent_index.values) 

    for nr_subj in range(11):
        drop_list = [('EXP_INFO', 'S{0}'.format(str(i_subj).zfill(2))) for i_subj in range(11) if i_subj != nr_subj]
        ind = np.arange(400)
        np.random.shuffle(ind)
        ind_shape = ind.reshape([16, -1])
        subj_df_list = []
        for nr_ses in range(4):
            with_ses = []
            ses_df_list = []
            for nr_block in range(4):
                select_ind = ind_shape[nr_ses*4 + nr_block]
                within_block = np.concatenate((last_pi[select_ind], mid_pi[select_ind]))
                np.random.shuffle(within_block)
                with_ses.append(within_block)
                
                exp_info = ['Ses_{0}, Block_{1}, Q_ind_{2}'.format(nr_ses, nr_block, i) for i in range(50)]
                all_beep_df.loc[within_block - 1, ('EXP_INFO', 'S{0}'.format(str(nr_subj).zfill(2)))] = exp_info

                ses_df_list.append(all_beep_df.copy().iloc[within_block - 1 ])
            ses_df = pd.concat(ses_df_list, ignore_index=True)
            ses_df.drop(columns=drop_list, inplace=True)
            ses_df.to_pickle('qa_info/S{0}_Session{1}_unshattered_beep_df.pkl'.format(str(nr_subj).zfill(2), str(nr_ses)))

            subj_df_list.append(ses_df)
            del ses_df
        subj_df = pd.concat(subj_df_list, ignore_index=True)
        subj_df.sort_values(('SENTENCE_INFO', 'permanent_index'), ascending=True, inplace=True)
        subj_df.to_pickle('qa_info/S{0}_unshattered_beep_df.pkl'.format(str(nr_subj).zfill(2)))    
        del subj_df
    all_beep_df.to_pickle('qa_info/all_unshattered_beep_df_randomized.pkl')     



def intro_run(win, obj, name):
    print('Log: ' + name + ' start')

    breakpoint_logger(comp=name, value=1, run=None, block=None, trial=None)
    # ------Prepare to start Routine "instruction"-------
    # update component parameters for each repeat
    if 'key_resp' in obj.keys():
        obj['key_resp'].keys = []
        obj['key_resp'].rt = []
    # keep track of which components have finished
    win, obj, objComponents, t, frameN, continueRoutine = pre_run_comp(win, obj)
    trigger_mat = np.zeros((len(objComponents) - 1, 2))
    comp_list = np.asarray([*obj['time'].keys()])
    # trigger_encoding_sending('instruction', input_run=0, input_block=0, intro_rec=0, input_event=0)
    # -------Run Routine "instruction"-------
    enumerate_list = [*instruction.keys()]
    for current_comp in enumerate_list.remove('clock'):
        if current_comp != 'key_resp':
            win, obj[current_comp], trigger_mat[0] = run_comp(
            win, obj[current_comp], 'text', frameN, t, tThisFlip, tThisFlipGlobal, 
            start_time=instruction['time']['text'][0], duration=instruction['time']['text'][1])

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

