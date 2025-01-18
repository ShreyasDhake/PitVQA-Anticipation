
import os
import pandas as pd

from utils_anticipation import (step_phase_mapping, next_phase_mapping,
                   step_operation_mapping, number_location_mapping,instrument_mapping)

count_dict = {}
frames = 0
questions = 0


def prepare_what_phase_qa(step_name):
    #step_name = step_name.replace('_', ' ').strip() 
    question = 'What is the surgical phase shown in the image?'
    answer = step_phase_mapping[step_name]
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa, answer


def prepare_what_step_qa(step_name):
    #step_name = step_name.replace('_', ' ').strip() 
    question = 'What is the surgical step shown in this frame?'
    answer = step_name
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa


def prepare_next_phase_qa(current_phase):
    #current_phase = current_phase.replace('_', ' ').strip()  
    question = 'What is the next surgical phase?'
    answer = next_phase_mapping[current_phase]
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
    question = 'What is the next surgical step?'
    answer = remaining_steps[0].split()[0]  
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
        answer = 'zero'
    elif num_of_tool == 1:
        answer = 'one'
    elif num_of_tool == 2:
        answer = 'two'
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa


def prepare_what_operation_qa(step_name):
    #step_name = step_name.replace('_', ' ').strip()     
    question = 'What is the surgical operation performed in the image?'
    answer = step_operation_mapping[step_name]
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
    time_left = None
    instrument_time = instrument_data.iloc[index]['int_time']

    # Sum the durations of repeated steps
    step_durations = {}
    for i in range(len(step_data) - 1):
        step_name = step_data.iloc[i]['str_step']
        step_start_time = step_data.iloc[i]['int_time']
        step_end_time = step_data.iloc[i + 1]['int_time']
        duration = step_end_time - step_start_time

        if step_name not in step_durations:
            step_durations[step_name] = 0
        step_durations[step_name] += duration
    #print(step_durations)
    # Calculate the remaining time for the current step
    for i in range(len(step_data) - 1):
        step_name = step_data.iloc[i]['str_step']
        step_start_time = step_data.iloc[i]['int_time']
        step_end_time = step_data.iloc[i + 1]['int_time']

        if step_start_time <= instrument_time < step_end_time:
            total_duration = step_durations[step_name]
            elapsed_time = instrument_time - step_start_time
            time_left = (total_duration - elapsed_time) / 60  # Convert seconds to minutes
            break

    if time_left is None:
        # Handle the case where the instrument time is greater than all step times
        time_left = 0
    time_left = float(f"{time_left:.3g}")
    return time_left

def prepare_step_time(time_left):
    question = 'How long is left for the current step?'
    answer = 'About ' + str(time_left) +' minutes left for the current step'
    qa = question + '|' + answer
    if answer not in count_dict.keys():
        count_dict[answer] = 1
    else:
        count_dict[answer] += 1
    return qa

def calculate_phase_durations(step_data,instrument_data, index,current_phase):
    nasalsphenoid_durations = []
    sellar_durations = []
    closure_durations = []
   
    for i in range(len(step_data['int_time'])-1):
        phase = current_phase

        if phase == 'nasal_sphenoid':
            nasalsphenoid_durations.append(step_data.iloc[i+1]['int_time']-step_data.iloc[i]['int_time'])
            #print(nasalsphenoid_durations)
        elif phase == 'sellar':
            sellar_durations.append(step_data.iloc[i+1]['int_time']-step_data.iloc[i]['int_time'])
        elif phase == 'closure':
            closure_durations.append(step_data.iloc[i+1]['int_time']-step_data.iloc[i]['int_time'])
        else:
            pass

    if phase == 'nasal_sphenoid':
        
        time_left = sum(nasalsphenoid_durations) - (instrument_data.iloc[index+1]['int_time']-instrument_data.iloc[index]['int_time'] ) 
    elif phase == 'sellar':
        time_left = sum(sellar_durations) - (instrument_data.iloc[index+1]['int_time']-instrument_data.iloc[index]['int_time'] ) 
        #print(time_left)
    elif phase == 'closure':
        time_left = sum(closure_durations) - (instrument_data.iloc[index+1]['int_time']-instrument_data.iloc[index]['int_time'] ) 
    else:
        time_left = 0
    time_left = float(f"{time_left/60:.3g}")
    #print(time_left)
    return time_left
    
def prepare_phase_time(time_left):
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
    question = 'What are the remaining surgical phases?'
    remaining_phases = []
    phase = step_phase_mapping[step_name]
    #print(phase)
    if phase == 'nasal_sphenoid':
        remaining_phases = ['sellar', 'closure']
    elif phase == 'sellar':
        remaining_phases = ['closure']
    elif phase == 'closure':
        remaining_phases = ['UNDEFINED PHASE: the step is not 1 of 14 steps.']
    elif phase == 'operation not started' or phase == 'operation ended' or phase == 'out of patient':
        remaining_phases = ['UNDEFINED PHASE: the step is not 1 of 14 steps.']
    else:
        remaining_phases = ['Unknown phase']
    
    answer = 'The remaining surgical phases are ' + ', '.join(remaining_phases)
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
    
    # Find the current step in the list
    for i in range(len(step_data)):
        if step_data.iloc[i]['int_time'] <= current_point < step_data.iloc[i + 1]['int_time']:
            for step in step_list[i+1:]:
                if step not in seen_steps:
                    remaining_steps.append(step)
                    seen_steps.add(step)
            break

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
            question = 'What instrument is used at the centre of the image?'
            answer = row_content['str_instrument1']
            qa = question + '|' + answer
            if answer not in count_dict.keys():
                count_dict[answer] = 1
            else:
                count_dict[answer] += 1
        else:
            position = number_location_mapping[str(row_content['pos_instrument1'])]
            question = f'What instrument is used in the {position} of the image?'
            answer = row_content['str_instrument1']
            qa = question + '|' + answer
            if answer not in count_dict.keys():
                count_dict[answer] = 1
            else:
                count_dict[answer] += 1
    elif pd.notna(row_content['pos_instrument2']):
        if str(row_content['pos_instrument2']) == '3.0':
            question = 'What instrument is used at the centre of the image?'
            answer = row_content['str_instrument2']
            qa = question + '|' + answer
            if answer not in count_dict.keys():
                count_dict[answer] = 1
            else:
                count_dict[answer] += 1
        else:
            position = number_location_mapping[str(row_content['pos_instrument2'])]
            question = f'What instrument is used in the {position} of the image?'
            answer = row_content['str_instrument2']
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
        question = f'What surgical activity is performing by the instrument {tool}?'
        answer = operation
        qa = question + '|' + answer
        if answer not in count_dict.keys():
            count_dict[answer] = 1
        else:
            count_dict[answer] += 1
    elif pd.notna(row_content['pos_instrument2']):
        tool = row_content['str_instrument2']
        question = f'What surgical activity is performing by the instrument {tool}?'
        answer = operation
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
        question = f'Where is the surgical instrument {tool} tip located in the image?'
        answer = number_location_mapping[str(row_content['pos_instrument1'])]
        qa = question + '|' + answer
        if answer not in count_dict.keys():
            count_dict[answer] = 1
        else:
            count_dict[answer] += 1
    elif pd.notna(row_content['pos_instrument2']):
        tool = row_content['str_instrument2']
        question = f'Where is the surgical instrument {tool} tip located in the image?'
        answer = number_location_mapping[str(row_content['pos_instrument2'])]
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
                question = 'What instrument is used at the centre of the image?'
                answer = row_content['str_instrument1']
                qa = question + '|' + answer
                if answer not in count_dict.keys():
                    count_dict[answer] = 1
                else:
                    count_dict[answer] += 1
            else:
                position = number_location_mapping[str(row_content['pos_instrument1'])]
                question = f'What instrument is used in the {position} of the image?'
                answer = row_content['str_instrument1']
                qa = question + '|' + answer
                if answer not in count_dict.keys():
                    count_dict[answer] = 1
                else:
                    count_dict[answer] += 1
    elif col_num == 2:
        if pd.notna(row_content['pos_instrument2']):
            if str(row_content['pos_instrument2']) == '3.0':
                question = 'What instrument is used at the centre of the image?'
                answer = row_content['str_instrument2']
                qa = question + '|' + answer
                if answer not in count_dict.keys():
                    count_dict[answer] = 1
                else:
                    count_dict[answer] += 1
            else:
                position = number_location_mapping[str(row_content['pos_instrument2'])]
                question = f'What instrument is used in the {position} of the image?'
                answer = row_content['str_instrument2']
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
            tool = row_content['str_instrument1']
            question = f'What surgical activity is performing by the instrument {tool}?'
            answer = operation
            qa = question + '|' + answer
            if answer not in count_dict.keys():
                count_dict[answer] = 1
            else:
                count_dict[answer] += 1
    elif col_num == 2:
        if pd.notna(row_content['pos_instrument2']):
            tool = row_content['str_instrument2']
            question = f'What surgical activity is performing by the instrument {tool}?'
            answer = operation
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
            tool = row_content['str_instrument1']
            question = f'Where is the surgical instrument {tool} tip located in the image?'
            answer = number_location_mapping[str(row_content['pos_instrument1'])]
            qa = question + '|' + answer
            if answer not in count_dict.keys():
                count_dict[answer] = 1
            else:
                count_dict[answer] += 1
    elif col_num == 2:
        if pd.notna(row_content['pos_instrument2']):
            tool = row_content['str_instrument2']
            question = f'Where is the surgical instrument {tool} tip located in the image?'
            answer = number_location_mapping[str(row_content['pos_instrument2'])]
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
    for i in [21]:
        print(f'processing file {i}')
        QA_folder = r"C:\Shreyas\Research Project\PitVQA_Anticipation\Test\qa"
        video_num = 'video_' + f"{i:03d}"
        video_folder = os.path.join(QA_folder, video_num)

        instrument_folder = r"C:\Shreyas\Research Project\PitVQA_Anticipation\Test\Instruments"
        instrument_num = 'instruments_' + f"{i:03d}" + '.csv'
        instrument_file = os.path.join(instrument_folder, instrument_num)

        step_folder = r"C:\Shreyas\Research Project\PitVQA_Anticipation\Test\Steps"
        step_num = 'steps_' + f"{i:03d}" + '.csv'
        step_file = os.path.join(step_folder, step_num)

        # open instrument file
        instrument_df = pd.read_csv(instrument_file)
        # open step file
        step_df = pd.read_csv(step_file)
        int_time_list = list(step_df['int_time'])

        # go through video folder
        for idx, file_name in enumerate(os.listdir(video_folder)):  # 遍历video_01下的所有txt文件; 0, 00000.txt; 自动忽略第一行
            # instrument csv file
            instrument_row = instrument_df.iloc[idx]  # 第idx+1行
            num_of_tool = get_num_of_tool(instrument_row)

            if (instrument_row['str_instrument1'] == 'out_of_patient' or
                    instrument_row['str_instrument2'] == 'out_of_patient'):
                continue

            # step csv file
            step_name = get_current_step_name(step_df, int_time_list, idx)
            if step_name in ['out_of_patient', 'operation_not_started', 'operation_ended']:
                continue
            
            print(f"Index: {idx}, DataFrame length: {len(instrument_df)}")
            if num_of_tool == 0:

                if num_of_tool not in tool_num:
                    tool_num[num_of_tool] = 1
                else:
                    tool_num[num_of_tool] += 1

                phase_qa_str, current_phase = prepare_what_phase_qa(step_name)  # 3
                step_qa_str = prepare_what_step_qa(step_name)  # 14

                next_phase_qa_str = prepare_next_phase_qa(current_phase)
                next_step_qa_str = prepare_next_step_qa(step_df, instrument_df, idx)

                how_many_tool_qa_str = prepare_how_many_tool_qa(num_of_tool)

                prepare_next_phase_str = prepare_next_phase_list(step_name)
                phase_time = calculate_phase_durations(step_df,instrument_df,idx,current_phase)
                prepare_next_phase_time_str = prepare_phase_time(phase_time)

                step_time = calculate_step_durations(step_df,instrument_df,idx)
                prepare_step_time_str = prepare_step_time(step_time)    

                overall_time = overall_time_calculations(step_df, instrument_df, idx)
                prepare_remaining_time_str = prepare_remaining_time(overall_time)

                next_step_str = prepare_next_step_list(step_df,instrument_df,idx)

                
                qa_list = [phase_qa_str, step_qa_str, next_phase_qa_str, next_step_qa_str, how_many_tool_qa_str,prepare_next_phase_str,prepare_next_phase_time_str,
                           prepare_step_time_str,prepare_remaining_time_str,next_step_str]
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
                next_step_qa_str = prepare_next_step_qa(step_df, instrument_df, idx)

                what_tool_qa_str = prepare_what_tool_qa_one(instrument_row)  # 0没有的
                tool_operation_qa_str = prepare_tool_operation_qa(instrument_row, operation)
                where_tool_qa_str = prepare_where_tool_qa(instrument_row)

                how_many_tool_qa_str = prepare_how_many_tool_qa(num_of_tool)

                prepare_next_phase_str = prepare_next_phase_list(step_name)
                phase_time = calculate_phase_durations(step_df,instrument_df,idx,current_phase)

                prepare_next_phase_time_str = prepare_phase_time(phase_time)

                step_time = calculate_step_durations(step_df,instrument_df,idx)
                prepare_step_time_str = prepare_step_time(step_time)

                overall_time = overall_time_calculations(step_df, instrument_df, idx)
                prepare_remaining_time_str = prepare_remaining_time(overall_time)

                next_step_str = prepare_next_step_list(step_df,instrument_df,idx)

                
                qa_list = [phase_qa_str, step_qa_str, operation_qa_str, next_phase_qa_str, next_step_qa_str,
                           what_tool_qa_str, tool_operation_qa_str, where_tool_qa_str, how_many_tool_qa_str,prepare_next_phase_str,prepare_next_phase_time_str,
                           prepare_step_time_str,prepare_remaining_time_str,next_step_str]
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
                next_step_qa_str = prepare_next_step_qa(step_df, instrument_df, idx)

                what_tool_qa_str_1 = prepare_what_tool_qa_two(instrument_row, col_num=1)
                what_tool_qa_str_2 = prepare_what_tool_qa_two(instrument_row, col_num=2)

                tool_operation_qa_str_1 = prepare_tool_operation_qa_two(instrument_row, operation, col_num=1)
                tool_operation_qa_str_2 = prepare_tool_operation_qa_two(instrument_row, operation, col_num=2)

                where_tool_qa_str_1 = prepare_where_tool_qa_two(instrument_row, col_num=1)
                where_tool_qa_str_2 = prepare_where_tool_qa_two(instrument_row, col_num=2)

                how_many_tool_qa_str = prepare_how_many_tool_qa(num_of_tool)

                prepare_next_phase_str = prepare_next_phase_list(step_name)

                phase_time = calculate_phase_durations(step_df,instrument_df,idx,current_phase)
                prepare_next_phase_time_str = prepare_phase_time(phase_time)

                step_time = calculate_step_durations(step_df,instrument_df,idx)
                prepare_step_time_str = prepare_step_time(step_time)

                overall_time = overall_time_calculations(step_df, instrument_df, idx)
                prepare_remaining_time_str = prepare_remaining_time(overall_time)
            
                next_step_str = prepare_next_step_list(step_df,instrument_df,idx)
                
                
                qa_list = [phase_qa_str, step_qa_str, operation_qa_str, next_phase_qa_str, next_step_qa_str,
                           what_tool_qa_str_1, what_tool_qa_str_2, tool_operation_qa_str_1, tool_operation_qa_str_2,
                           where_tool_qa_str_1, where_tool_qa_str_2, how_many_tool_qa_str,prepare_next_phase_str,prepare_next_phase_time_str,
                           prepare_step_time_str,prepare_remaining_time_str,next_step_str]
                write_file(video_folder, file_name, qa_list)
                questions += 12
                frames += 1

        print(count_dict)
        print(f'video_{i} finished.')

    print(tool_num)

    print(f'number of frames: {frames}')
    print(f'number of questions: {questions}')
    print(f'sum of dict: {sum(count_dict.values())}')
