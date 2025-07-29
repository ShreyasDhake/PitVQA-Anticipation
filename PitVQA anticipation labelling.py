
import os
import pandas as pd

from utils_anticipation import (step_phase_mapping, next_phase_mapping,
                   step_operation_mapping, number_location_mapping,instrument_mapping)

count_dict = {}
frames = 0
questions = 0


def prepare_what_phase_qa(step_name):
    #step_name = step_name.replace('_', ' ').strip() 
    question = 'What is the phase shown in the image?'
    canswer = step_phase_mapping[step_name]
    phase = canswer.replace('_', ' ').strip()
    answer = 'The phase shown in the image is '+str(phase)
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa, canswer


def prepare_what_step_qa(step_name):
    #step_name = step_name.replace('_', ' ').strip() 
    question = 'What is the step shown in the image?'
    answer = step_name
    answer = answer.replace('_', ' ').strip()
    answer = 'The step shown in the image is '+str(answer)
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa


def prepare_next_phase_qa(current_phase):
    #current_phase = current_phase.replace('_', ' ').strip()  
    question = 'What is the next phase?'
    phase = next_phase_mapping[current_phase]
    phase = phase.replace('_', ' ').strip()
    answer = 'The next phase is '+str(phase)
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa


def prepare_next_step_qa(step_data, instrument_data, index):
    current_point = instrument_data.iloc[index]['int_time']
    remaining_steps = []
    seen_steps = set()
    step_list = step_data['str_step']
    
    # Find the current step in the list
    for i in range(len(step_data)):
        if step_data.iloc[i]['int_time'] <= current_point < step_data.iloc[i + 1]['int_time']:
            for step in step_list[i+1:]:
                if step not in seen_steps:
                    remaining_steps.append(step)
                    seen_steps.add(step)
            break
    question = 'What is the next step?'
    answer = remaining_steps[0].split()[0]
    answer = answer.replace('_', ' ').strip()
    answer = 'The next step is '+str(answer)  
    if answer == 'operation_not_started' or answer == 'operation_ended' or answer == 'out_of_patient':
        answer = 'UNDEFINED STEP: the step is not 1 of 14 steps.'
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa


def prepare_how_many_tool_qa(num_of_tool):
    question = 'How many instruments are present in the image?'
    answer = num_of_tool
    if num_of_tool == 0:
        answer = 'Zero instruments are present in the image'
    elif num_of_tool == 1:
        answer = 'One instruments are present in the image'
    elif num_of_tool == 2:
        answer = 'Two instruments are present in the image'
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa


def prepare_what_operation_qa(step_name):
    #step_name = step_name.replace('_', ' ').strip()     
    question = 'What is the surgical activity in the image?'
    answer = 'The surgical activity in the image is '+step_operation_mapping[step_name]
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa, answer

def overall_time_calculations(step_data, instrument_data, index):
    end_point = step_data.iloc[-1]['int_time']
    #print(end_point)
    current_time = instrument_data.iloc[index]['int_time']
    time_left = end_point - current_time
    time_left = float(f"{time_left/60:.3g}")
    return time_left

def prepare_remaining_time(time_left):
    question = 'What is the estimated remaining time for the entire surgery?'
    answer = 'The estimated remaining time for the entire surgery is ' + str(time_left) + ' minutes'
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa

def calculate_step_durations(step_data, instrument_data, index):
    instrument_time = instrument_data.iloc[index]['int_time']
    step_name = None
    # Find the current step and interval
    for i in range(len(step_data) - 1):
        start = step_data.iloc[i]['int_time']
        end = step_data.iloc[i + 1]['int_time']
        if start <= instrument_time < end:
            step_name = step_data.iloc[i]['str_step']
            current_interval_idx = i
            break

    if step_name is None:
        return 0.0

    # Remaining time in current interval
    start = step_data.iloc[current_interval_idx]['int_time']
    end = step_data.iloc[current_interval_idx + 1]['int_time']
    time_left = end - instrument_time

    # Add durations of all future intervals of the same step
    for i in range(current_interval_idx + 1, len(step_data) - 1):
        if step_data.iloc[i]['str_step'] == step_name:
            time_left += step_data.iloc[i + 1]['int_time'] - step_data.iloc[i]['int_time']

    time_left = float(f"{time_left/60:.3g}")
    return time_left

# def calculate_step_durations(step_data, instrument_data, index):
#     instrument_time = instrument_data.iloc[index]['int_time']
#     time_left = 0
#     for i in range(len(step_data) - 1):
#         step_start_time = step_data.iloc[i]['int_time']
#         step_end_time = step_data.iloc[i + 1]['int_time']
#         if step_start_time <= instrument_time < step_end_time:
#             time_left = (step_end_time - instrument_time) / 60  # seconds to minutes
#             break
#     time_left = float(f"{time_left:.3g}")
#     return time_left

def prepare_step_time(time_left):
    question = 'How long is left for the current step?'
    answer = 'About ' + str(time_left) +' minutes left for the current step'
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa

def calculate_phase_durations(step_data):
    nasalsphenoid_durations = []
    sellar_durations = []
    closure_durations = []
   
    for i in range(len(step_data['int_time'])-1):
        phase = step_phase_mapping[step_data.iloc[i]['str_step']]

        if phase == 'nasal sphenoid':
            nasalsphenoid_durations.append(step_data.iloc[i+1]['int_time']-step_data.iloc[i]['int_time'])
            #print(nasalsphenoid_durations)
        elif phase == 'sellar':
            sellar_durations.append(step_data.iloc[i+1]['int_time']-step_data.iloc[i]['int_time'])
        elif phase == 'closure':
            closure_durations.append(step_data.iloc[i+1]['int_time']-step_data.iloc[i]['int_time'])
        else:
            pass
    nasalsphenoid_durations_ = sum(nasalsphenoid_durations) 
    sellar_durations_ = sum(sellar_durations)
    closure_durations_ = sum(closure_durations)
    #print(nasalsphenoid_durations_)
    #print(sellar_durations_)
    #print(closure_durations_)
    return nasalsphenoid_durations_, sellar_durations_, closure_durations_

# def prepare_phase_time(nasalsphenoid_durations, sellar_durations, closure_durations,instrument_data,index,phase):

#     if phase == 'nasal sphenoid':
#         time_left = nasalsphenoid_durations - (instrument_data.iloc[index+1]['int_time']-instrument_data.iloc[index]['int_time'] ) 
#     elif phase == 'sellar':
#         time_left = sellar_durations - (instrument_data.iloc[index+1]['int_time']-instrument_data.iloc[index]['int_time'] ) 
#         #print(time_left)
#     elif phase == 'closure':
#         time_left = closure_durations - (instrument_data.iloc[index+1]['int_time']-instrument_data.iloc[index]['int_time'] ) 
#     else:
#         time_left = 0
#     time_left = float(f"{time_left/60:.3g}")
#     #print(time_left)
    
#     question = 'How long is left for the current phase?'
#     answer = 'About ' + str(time_left) +' minutes left for the current phase'
#     qa = question + '|' + answer
#     if answer not in count_dict.keys():
#         count_dict[answer] = 1
#     else:
#         count_dict[answer] += 1
#     return qa

def prepare_phase_time(nasalsphenoid_durations, sellar_durations, closure_durations, instrument_data, index, phase, step_df):
    current_time = instrument_data.iloc[index]['int_time']
    # 1. Get all intervals for this phase
    intervals = []
    for i in range(len(step_df)-1):
        step_phase = step_phase_mapping[step_df.iloc[i]['str_step']]
        if step_phase == phase:
            step_start = step_df.iloc[i]['int_time']
            step_end = step_df.iloc[i+1]['int_time']
            intervals.append((step_start, step_end))
    # 2. Calculate elapsed time in this phase up to current_time
    elapsed = 0
    for start, end in intervals:
        if current_time >= end:
            elapsed += end - start
        elif start <= current_time < end:
            elapsed += current_time - start
            break
    # 3. Get total duration for this phase
    if phase == 'nasal sphenoid':
        total = nasalsphenoid_durations
    elif phase == 'sellar':
        total = sellar_durations
    elif phase == 'closure':
        total = closure_durations
    else:
        total = 0
    # 4. Calculate time left
    time_left = (total - elapsed) / 60
    time_left = float(f"{time_left:.3g}")

    question = 'How long is left for the current phase?'
    answer = 'About ' + str(time_left) +' minutes left for the current phase'
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa


def prepare_next_phase_list(step_name):
    #step_name = step_name.replace('_', ' ').strip()     
    question = 'What are the remaining phases?'
    remaining_phases = []
    phase = step_phase_mapping[step_name]
    #print(phase)
    if phase == 'nasal sphenoid':
        remaining_phases = ['sellar', 'closure']
    elif phase == 'sellar':
        remaining_phases = ['closure']
    elif phase == 'closure':
        remaining_phases = ['UNDEFINED PHASE: the step is not 1 of 14 steps.']
    elif phase == 'operation not started' or phase == 'operation ended' or phase == 'out of patient':
        remaining_phases = ['UNDEFINED PHASE: the step is not 1 of 14 steps.']
    else:
        remaining_phases = ['Unknown phase']
    
    answer = 'The remaining phases are ' + ', '.join(remaining_phases)
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa


# def instrument_for_next_step( index, df_instrument1, df_instrument2):
#     instr1 = instrument_mapping[df_instrument1[index+1]]
#     instr2 = instrument_mapping[df_instrument2[index+1]]
#     invalid = {'out of patient',
#                'no secondary instrument',
#                'no visible instrument'}

#     # only pick instr1 if instr2 is “invalid” but instr1 is not
#     if instr2 in invalid and instr1 not in invalid:
#         next_instruments = instr1
#     else:
#         if instr1 == 'out of patient':
#             next_instruments = 'out of patient'
#         elif instr1 == 'no visible instrument':
#             next_instruments = 'no visible instrument'
#         else:
#             next_instruments = instr1 + ' and ' + instr2
#     question = 'What are the instruments required for the next step?'
#     answer = 'The instruments required for the next step are: ' + (next_instruments)
#     qa = question + '|' + answer
#     if answer not in count_dict.keys():
#         count_dict[answer] = 1
#     else:
#         count_dict[answer] += 1
#     return qa
def instrument_for_next_step(index, df_instrument1, df_instrument2):
    instr1 = instrument_mapping[df_instrument1[index+1]]
    instr2 = instrument_mapping[df_instrument2[index+1]]
    invalid = {'out of patient',
               'no secondary instrument',
               'no visible instrument',
               'suction'}  # Added 'suction' to invalid instruments

    # Filter out suction instruments by treating them as invalid
    if instr1 == 'suction':
        instr1 = 'no visible instrument'
    if instr2 == 'suction':
        instr2 = 'no secondary instrument'

    # Logic to determine next instruments
    if instr2 in invalid and instr1 not in invalid:
        next_instruments = instr1
    elif instr1 in invalid and instr2 not in invalid:
        next_instruments = instr2
    elif instr1 in invalid and instr2 in invalid:
        # Both instruments are invalid (including suction), return generic message
        next_instruments = 'no visible instrument'
    else:
        # Both instruments are valid (non-suction, non-invalid)
        next_instruments = instr1 + ' and ' + instr2
    
    question = 'What are the instruments required for the next step?'
    answer = 'The instruments required for the next step are: ' + next_instruments
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa


def prepare_next_step_list(step_data, instrument_data, index):
    current_point = instrument_data.iloc[index]['int_time']
    remaining_steps = []
    seen_steps = set()
    step_list = step_data['str_step']
    #print(step_list)
    # Find the current step in the list
    #print(len(step_data))
    #print(step_data)
    for i in range(len(step_data)):
        #print(len(step_data))
        if step_data.iloc[i]['int_time'] <= current_point < step_data.iloc[i + 1]['int_time']:
            #print(step_data.iloc[i]['int_time'])
            #print(step_data.iloc[i + 1]['int_time'])
            for step in step_list[i+1:]:
                if step not in seen_steps:
                    step = step.replace('_', ' ').strip()
                    remaining_steps.append(step)
                    seen_steps.add(step)
            

    question = 'What are the remaining steps?'
    answer = 'The remaining steps are: ' + ', '.join(remaining_steps)
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa




def prepare_what_tool_qa_one(row_content):
    # What instrument is used in the bottom-mid of the image? | 1 of 18 instruments
    qa = 0
    if pd.notna(row_content['pos_instrument1']):
        if str(row_content['pos_instrument1']) == '3.0':
            question = 'What instrument is present in the centre of the image?'
            name = row_content['str_instrument1'].replace('_', ' ').strip()
            answer = 'The instrument present in the centre of the image is '+name
            qa = question + '|' + answer
            if answer not in count_dict.keys():
                count_dict[answer] = 1
            else:
                count_dict[answer] += 1
        else:
            position = number_location_mapping[str(row_content['pos_instrument1'])]
            question = f'What instrument is present in the {position} of the image?'
            answer = 'The instrument present in the '+position+' of the image is '+row_content['str_instrument1'].replace('_', ' ').strip()
            qa = question + '|' + answer
            if answer not in count_dict.keys():
                count_dict[answer] = 1
            else:
                count_dict[answer] += 1
    elif pd.notna(row_content['pos_instrument2']):
        if str(row_content['pos_instrument2']) == '3.0':
            question = 'What instrument is present in the centre of the image?'
            answer = 'The instrument present in the centre of the image is '+row_content['str_instrument2'].replace('_', ' ').strip()
            qa = question + '|' + answer
            if answer not in count_dict.keys():
                count_dict[answer] = 1
            else:
                count_dict[answer] += 1
        else:
            position = number_location_mapping[str(row_content['pos_instrument2'])]
            question = f'What instrument is used in the {position} of the image?'
            answer = 'The instrument present in the '+position+' of the image is '+row_content['str_instrument2'].replace('_', ' ').strip()
            qa = question + '|' + answer
            if answer not in count_dict.keys():
                count_dict[answer] = 1
            else:
                count_dict[answer] += 1
    return qa


def prepare_tool_operation_qa(row_content, operation):
    # What surgical activity is performing by the instrument kerrisons? | 1 of 14 operations
    qa = 0
    if pd.notna(row_content['pos_instrument1']):
        tool = row_content['str_instrument1']
        operation = operation.replace('_', ' ').strip()
        tool = tool.replace('_', ' ').strip()
        question = 'What is the surgical activity in the image?'
        answer = 'The surgical activity in the image is '+operation
        qa = question + '|' + answer
        if answer not in count_dict.keys():
            count_dict[answer] = 1
        else:
            count_dict[answer] += 1
    elif pd.notna(row_content['pos_instrument2']):
        tool = row_content['str_instrument2']
        operation = operation.replace('_', ' ').strip()
        tool = tool.replace('_', ' ').strip()
        question = 'What is the surgical activity in the image?'
        answer = 'The surgical activity in the image is '+operation
        qa = question + '|' + answer
        if answer not in count_dict.keys():
            count_dict[answer] = 1
        else:
            count_dict[answer] += 1
    return qa


def prepare_where_tool_qa(row_content):
    # Where is the surgical instrument kerrisons tip located in the image? | 1 of 5 locations
    qa = 0
    if pd.notna(row_content['pos_instrument1']):
        tool = row_content['str_instrument1']
        tool = tool.replace('_', ' ').strip()
        question = f'Where is the {tool} located in the image?'
        answer = number_location_mapping[str(row_content['pos_instrument1'])]
        answer = f'The {tool} is located in the '+answer+' of the image.'
        qa = question + '|' + answer
        if answer not in count_dict.keys():
            count_dict[answer] = 1
        else:
            count_dict[answer] += 1
    elif pd.notna(row_content['pos_instrument2']):
        tool = row_content['str_instrument2']
        tool = tool.replace('_', ' ').strip()
        question = f'Where is the {tool} located in the image?'
        answer = number_location_mapping[str(row_content['pos_instrument2'])]
        answer = f'The {tool} is located in the '+answer+' of the image.'
        qa = question + '|' + answer
        if answer not in count_dict.keys():
            count_dict[answer] = 1
        else:
            count_dict[answer] += 1
    return qa


def prepare_what_tool_qa_two(row_content, col_num):
    # What instrument is used in the bottom-mid of the image? | 1 of 18 instruments
    qa = 0
    if col_num == 1:  # tool 1
        if pd.notna(row_content['pos_instrument1']):
            if str(row_content['pos_instrument1']) == '3.0':
                question = 'What instrument is present in the centre of the image?'
                answer = 'The instrument present in the centre of the image is '+row_content['str_instrument1'].replace('_', ' ').strip()
                qa = question + '|' + answer
                if answer not in count_dict.keys():
                    count_dict[answer] = 1
                else:
                    count_dict[answer] += 1
            else:
                position = number_location_mapping[str(row_content['pos_instrument1'])]
                question = f'What instrument is present in the {position} of the image?'
                answer = f'The instrument present in the {position} of the image is '+row_content['str_instrument1'].replace('_', ' ').strip()
                qa = question + '|' + answer
                if answer not in count_dict.keys():
                    count_dict[answer] = 1
                else:
                    count_dict[answer] += 1
    elif col_num == 2:
        if pd.notna(row_content['pos_instrument2']):
            if str(row_content['pos_instrument2']) == '3.0':
                question = 'What instrument is present in the centre of the image?'
                answer = 'The instrument present in the centre of the image is '+row_content['str_instrument2'].replace('_', ' ').strip()
                qa = question + '|' + answer
                if answer not in count_dict.keys():
                    count_dict[answer] = 1
                else:
                    count_dict[answer] += 1
            else:
                position = number_location_mapping[str(row_content['pos_instrument2'])]
                question = f'What instrument is present in the {position} of the image?'
                answer = f'The instrument present in the {position} of the image is '+row_content['str_instrument2'].replace('_', ' ').strip()
                qa = question + '|' + answer
                if answer not in count_dict.keys():
                    count_dict[answer] = 1
                else:
                    count_dict[answer] += 1
    return qa


def prepare_tool_operation_qa_two(row_content, operation, col_num):
    # What surgical activity is performing by the instrument kerrisons? | 1 of 14 operations
    qa = 0
    if col_num == 1:
        if pd.notna(row_content['pos_instrument1']):
            question = 'What is the surgical activity in the image?'
            answer = 'The surgical activity in the image is '+operation
            qa = question + '|' + answer
            if answer not in count_dict.keys():
                count_dict[answer] = 1
            else:
                count_dict[answer] += 1
    elif col_num == 2:
        if pd.notna(row_content['pos_instrument2']):
            question = 'What is the surgical activity in the image?'
            answer = 'The surgical activity in the image is '+operation
            qa = question + '|' + answer
            if answer not in count_dict.keys():
                count_dict[answer] = 1
            else:
                count_dict[answer] += 1
    return qa


def prepare_where_tool_qa_two(row_content, col_num):
    # Where is the surgical instrument kerrisons tip located in the image? | 1 of 5 locations
    qa = 0
    if col_num == 1:
        if pd.notna(row_content['pos_instrument1']):
            tool = row_content['str_instrument1'].replace('_', ' ').strip()
            question = f'Where is the {tool} located in the image?'
            answer = number_location_mapping[str(row_content['pos_instrument1'])]
            answer = f'The {tool} is located in the '+answer+' of the image.'
            qa = question + '|' + answer
            if answer not in count_dict.keys():
                count_dict[answer] = 1
            else:
                count_dict[answer] += 1
    elif col_num == 2:
        if pd.notna(row_content['pos_instrument2']):
            tool = row_content['str_instrument2'].replace('_', ' ').strip()
            question = f'Where is the {tool} located in the image?'
            answer = number_location_mapping[str(row_content['pos_instrument2'])]
            answer = f'The {tool} is located in the '+answer+' of the image.'
            qa = question + '|' + answer
            if answer not in count_dict.keys():
                count_dict[answer] = 1
            else:
                count_dict[answer] += 1
    return qa


def get_num_of_tool(row_content):  # row_content is the content for that row
    if pd.isna(row_content['pos_instrument1']) and pd.isna(row_content['pos_instrument2']):
        number_of_tool = 0
    elif pd.notna(row_content['pos_instrument1']) and pd.notna(row_content['pos_instrument2']):
        number_of_tool = 2
    else:
        number_of_tool = 1
    return number_of_tool


def get_current_step_name(step_df, time_list, index):
    row_idx = 0
    for time in time_list:
        if index < time:
            row_idx = time_list.index(time)-1
            break
    step_name = step_df.iloc[row_idx]['str_step']
    return step_name


def write_file(video_folder, file_name, qa_list):
    file = os.path.join(video_folder, file_name)
    with open(file, 'w', encoding='utf-8') as f:
        for qa in qa_list:
            f.write(qa + '\n')


if __name__ == "__main__":

    tool_num = {}
    for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]:
    #for i in [1]:
        print(f'processing file {i}')
        QA_folder = r"C:\Shreyas\Research Project\PitVQA_Anticipation\Test\QA_new"
        video_num = 'video_' + f"{i:02d}"
        video_folder = os.path.join(QA_folder, video_num)

        instrument_folder = r"C:\Shreyas\Research Project\PitVQA_Anticipation\Test\Instruments2"
        instrument_num = 'instruments_' + f"{i:02d}" + '.csv'
        instrument_file = os.path.join(instrument_folder, instrument_num)

        step_folder = r"C:\Shreyas\Research Project\PitVQA_Anticipation\Test\Steps"
        step_num = 'steps_' + f"{i:02d}" + '.csv'
        step_file = os.path.join(step_folder, step_num)

        # open instrument file
        instrument_df = pd.read_csv(instrument_file)
        df_instrument1 = instrument_df['int_instrument1']
        df_instrument2 = instrument_df['int_instrument2']
        # open step file
        step_df = pd.read_csv(step_file)

        #print(step_df.to_string())

        int_time_list = list(step_df['int_time'])


        # go through video folder
        for idx, file_name in enumerate(os.listdir(video_folder)):  # 遍历video_01下的所有txt文件; 0, 00000.txt; 自动忽略第一行
            base = int(os.path.splitext(file_name)[0]) 
            #print(int(base)) 
            #print(idx)
            # instrument csv file
            instrument_row = instrument_df.iloc[idx]  # 第idx+1行
            num_of_tool = get_num_of_tool(instrument_row)

            # if (instrument_row['str_instrument1'] == 'out_of_patient' or
            #         instrument_row['str_instrument2'] == 'out_of_patient'):
            #     continue

            # # step csv file
            step_name = get_current_step_name(step_df, int_time_list, base)
            # if step_name in ['out_of_patient', 'operation_not_started', 'operation_ended']:
            #     continue
            nasalsphenoid_durations, sellar_durations, closure_durations = calculate_phase_durations(step_df)
            print(f"Index: {base}, DataFrame length: {len(instrument_df)}")
            if num_of_tool == 0:

                if num_of_tool not in tool_num:
                    tool_num[num_of_tool] = 1
                else:
                    tool_num[num_of_tool] += 1

                phase_qa_str, current_phase = prepare_what_phase_qa(step_name)  # 3
                step_qa_str = prepare_what_step_qa(step_name)  # 14

                next_phase_qa_str = prepare_next_phase_qa(current_phase)
                next_step_qa_str = prepare_next_step_qa(step_df, instrument_df, base)

                how_many_tool_qa_str = prepare_how_many_tool_qa(num_of_tool)

                prepare_next_phase_str = prepare_next_phase_list(step_name)
                prepare_phase_time_str = prepare_phase_time(nasalsphenoid_durations, sellar_durations, closure_durations,instrument_df,base,current_phase,step_df)

                step_time = calculate_step_durations(step_df,instrument_df,base)
                prepare_step_time_str = prepare_step_time(step_time)    

                overall_time = overall_time_calculations(step_df, instrument_df, base)
                prepare_remaining_time_str = prepare_remaining_time(overall_time)

                next_step_str = prepare_next_step_list(step_df,instrument_df,base)
                next_instrument_str = instrument_for_next_step(base, df_instrument1, df_instrument2)
                
                qa_list = [phase_qa_str, step_qa_str, next_phase_qa_str, next_step_qa_str, how_many_tool_qa_str,prepare_next_phase_str,
                              prepare_step_time_str,prepare_remaining_time_str, next_instrument_str,prepare_phase_time_str]
                # qa_list = [next_phase_qa_str, next_step_qa_str, prepare_next_phase_str,
                # prepare_step_time_str,prepare_remaining_time_str, next_instrument_str,prepare_phase_time_str]
                #qa_list = [prepare_step_time_str, prepare_phase_time_str, prepare_remaining_time_str]
                write_file(video_folder, file_name, qa_list)
                questions += 5
                frames += 1

            if num_of_tool == 1:

                if num_of_tool not in tool_num:
                    tool_num[num_of_tool] = 1
                else:
                    tool_num[num_of_tool] += 1

                phase_qa_str, current_phase = prepare_what_phase_qa(step_name)
                step_qa_str = prepare_what_step_qa(step_name)
                operation_qa_str, operation = prepare_what_operation_qa(step_name)  # 0没有的

                next_phase_qa_str = prepare_next_phase_qa(current_phase)
                next_step_qa_str = prepare_next_step_qa(step_df, instrument_df, base)

                what_tool_qa_str = prepare_what_tool_qa_one(instrument_row)  # 0没有的
                #tool_operation_qa_str = prepare_tool_operation_qa(instrument_row, operation)
                where_tool_qa_str = prepare_where_tool_qa(instrument_row)

                how_many_tool_qa_str = prepare_how_many_tool_qa(num_of_tool)

                prepare_next_phase_str = prepare_next_phase_list(step_name)
                prepare_phase_time_str = prepare_phase_time(nasalsphenoid_durations, sellar_durations, closure_durations,instrument_df,base,current_phase,step_df)


                step_time = calculate_step_durations(step_df,instrument_df,base)
                prepare_step_time_str = prepare_step_time(step_time)

                overall_time = overall_time_calculations(step_df, instrument_df, base)
                prepare_remaining_time_str = prepare_remaining_time(overall_time)

                next_step_str = prepare_next_step_list(step_df,instrument_df,base)
                next_instrument_str = instrument_for_next_step(base, df_instrument1, df_instrument2)

                
                qa_list = [phase_qa_str, step_qa_str, operation_qa_str, next_phase_qa_str, next_step_qa_str,
                              what_tool_qa_str, how_many_tool_qa_str,prepare_next_phase_str,
                              prepare_step_time_str,prepare_remaining_time_str,next_instrument_str,prepare_phase_time_str,]
                # qa_list = [next_phase_qa_str, next_step_qa_str, prepare_next_phase_str,
                # prepare_step_time_str,prepare_remaining_time_str, next_instrument_str,prepare_phase_time_str]
                #qa_list = [prepare_step_time_str, prepare_phase_time_str, prepare_remaining_time_str]
                write_file(video_folder, file_name, qa_list)
                questions += 9
                frames += 1

            if num_of_tool == 2:

                if num_of_tool not in tool_num:
                    tool_num[num_of_tool] = 1
                else:
                    tool_num[num_of_tool] += 1

                phase_qa_str, current_phase = prepare_what_phase_qa(step_name)
                step_qa_str = prepare_what_step_qa(step_name)
                operation_qa_str, operation = prepare_what_operation_qa(step_name)

                next_phase_qa_str = prepare_next_phase_qa(current_phase)
                next_step_qa_str = prepare_next_step_qa(step_df, instrument_df, base)

                what_tool_qa_str_1 = prepare_what_tool_qa_two(instrument_row, col_num=1)
                what_tool_qa_str_2 = prepare_what_tool_qa_two(instrument_row, col_num=2)

                #tool_operation_qa_str_1 = prepare_tool_operation_qa_two(instrument_row, operation, col_num=1)
                #tool_operation_qa_str_2 = prepare_tool_operation_qa_two(instrument_row, operation, col_num=2)

                where_tool_qa_str_1 = prepare_where_tool_qa_two(instrument_row, col_num=1)
                where_tool_qa_str_2 = prepare_where_tool_qa_two(instrument_row, col_num=2)

                how_many_tool_qa_str = prepare_how_many_tool_qa(num_of_tool)

                prepare_next_phase_str = prepare_next_phase_list(step_name)

                prepare_phase_time_str = prepare_phase_time(nasalsphenoid_durations, sellar_durations, closure_durations,instrument_df,base,current_phase,step_df)


                step_time = calculate_step_durations(step_df,instrument_df,base)
                prepare_step_time_str = prepare_step_time(step_time)

                overall_time = overall_time_calculations(step_df, instrument_df, base)
                prepare_remaining_time_str = prepare_remaining_time(overall_time)
            
                next_step_str = prepare_next_step_list(step_df,instrument_df,base)
                
                next_instrument_str = instrument_for_next_step(base, df_instrument1, df_instrument2)

                qa_list = [phase_qa_str, step_qa_str, operation_qa_str, next_phase_qa_str, next_step_qa_str,
                              what_tool_qa_str_1, what_tool_qa_str_2, how_many_tool_qa_str,prepare_next_phase_str,
                              prepare_step_time_str,prepare_remaining_time_str,next_instrument_str,prepare_phase_time_str]
                # qa_list = [next_phase_qa_str, next_step_qa_str, prepare_next_phase_str,
                # prepare_step_time_str,prepare_remaining_time_str, next_instrument_str,prepare_phase_time_str]
                #qa_list = [prepare_step_time_str, prepare_phase_time_str, prepare_remaining_time_str]
                write_file(video_folder, file_name, qa_list)
                questions += 12
                frames += 1

        print(count_dict)
        print(f'video_{i} finished.')

    print(tool_num)

    print(f'number of frames: {frames}')
    print(f'number of questions: {questions}')
    print(f'sum of dict: {sum(count_dict.values())}')

