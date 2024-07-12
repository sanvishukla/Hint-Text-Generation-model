import uiautomator2 as u2
import os
import json
import xmltodict
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pprint

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('/Users/sanvishukla/Desktop/SRIP/fine-tuned-model')
model = GPT2LMHeadModel.from_pretrained('/Users/sanvishukla/Desktop/SRIP/fine-tuned-model')
feedback_log = []
reward_threshold = 0 
def getAllComponents(jsondata: dict):
    root = jsondata['hierarchy']
    queue = [root]
    res = []
    while queue:
        currentNode = queue.pop(0)
        if 'node' in currentNode:
            if type(currentNode['node']).__name__ == 'dict':
                queue.append(currentNode['node'])
            else:
                for e in currentNode['node']:
                    queue.append(e)
        else:
            if ('com.android.systemui' not in currentNode['@resource-id']) and ('com.android.systemui' not in currentNode['@package']):
                res.append(currentNode)
    return res

def find_EditText(jsondata: dict):
    all_components = getAllComponents(jsondata)
    ans = []
    for e_component in all_components:
        if '@class' in e_component and (e_component['@class'] == 'android.widget.EditText' or e_component['@class'] == 'android.widget.AutoCompleteTextView'):
            ans.append(e_component)
    return ans

def get_basic_info(e_component: dict):
    key_list = ['id', 'text', 'label', 'text-hint', 'app_name']
    key_at_list = ['resource-id', 'text', 'label', 'content-desc', 'package']
    dict_info = {}
    for i in range(len(key_list)):
        dict_info[key_list[i]] = None
        for e_property in e_component:
            if key_at_list[i] in e_property.lower():
                dict_info[key_list[i]] = e_component[e_property]
                break
    return dict_info
def parse_bounds(bounds_str: str) -> list:
    bounds_str = bounds_str.strip('[]')  # Remove brackets
    left_top, right_bottom = bounds_str.split('][')
    left, top = map(int, left_top.split(','))
    right, bottom = map(int, right_bottom.split(','))
    return [left, top, right, bottom]

def chooseFromPos(all_components: list, bounds: str):
    bounds_list = parse_bounds(bounds)
    
    same_horizon_components = []
    same_vertical_components = []
    
    left, top, right, bottom = bounds_list
    
    for e_component in all_components:
        e_bounds_str = e_component['@bounds']
        e_bounds = parse_bounds(e_bounds_str)
        e_left, e_top, e_right, e_bottom = e_bounds
        
        if e_bounds == bounds_list:
            continue
        
        if e_bottom == top and e_left <= right and e_right >= left:
            same_vertical_components.append(e_component)
        
        if e_top == bottom and e_left <= right and e_right >= left:
            same_horizon_components.append(e_component)
    
    return same_horizon_components, same_vertical_components

def turn_null_to_str(prop: str):
    return '' if prop is None else prop

def component_basic_info(jsondata: dict):
    text_id = "The purpose of this component may be '<EditText id>'. "
    text_label = "The label of this component is '<label>'. "
    text_text = "The text on this component is '<text>'. "
    text_hint = "The hint text of this component is '<hint>'. "
    if jsondata['id'] in (None, ''):
        text_id = ""
    else:
        EditText_id = jsondata['id'].split('/')[-1].replace('_', ' ')
        text_id = text_id.replace('<EditText id>', EditText_id)
    if jsondata['label'] in (None, ''):
        text_label = ""
    else:
        text_label = text_label.replace('<label>', jsondata['label'])
    if jsondata['text'] in (None, ''):
        text_text = ""
    else:
        text_text = text_text.replace('<text>', jsondata['text'])
    if jsondata['text-hint'] in (None, ''):
        text_hint = ""
    else:
        text_hint = text_hint.replace('<hint>', jsondata['text-hint'])
    return text_id + text_label + text_text + text_hint + '\n'

def isEnglish(s: str):
    s = s.replace('\u2026', '')
    return s.isascii()

def use_context_info_generate_prompt(jsondata: dict):
    text_header = "Question: "
    text_app_name = "This is a <app name> app. "
    text_activity_name = "On its page, it has an input component. "
    text_label = "The label of this component is '<label>'. "
    text_text = "The text on this component is '<text>'. "
    text_context_info = "Below is the relevant prompt information of the input component:\n<context information>"
    text_id = "The purpose of this input component may be '<EditText id>'. "
    text_ask = "What is the hint text of this input component?\n"
    app_name = jsondata['app_name'].split('.')[-1]
    text_app_name = text_app_name.replace('<app name>', app_name)
    if jsondata['label'] in (None, ''):
        text_label = ""
    else:
        text_label = text_label.replace('<label>', jsondata['label'])
    if jsondata['text'] in (None, ''):
        text_text = ""
    else:
        text_text = text_text.replace('<text>', jsondata['text'])
    context_info = ""
    if len(jsondata['same-horizon']) > 0:
        for e in jsondata['same-horizon']:
            if not isEnglish(turn_null_to_str(e['label']) + turn_null_to_str(e['text']) + turn_null_to_str(e['text-hint'])):
                continue
            context_info += "There is a component on the same horizontal line as this input component. "
            context_info += component_basic_info(e)
    if len(jsondata['same-vertical']) > 0:
        for e in jsondata['same-vertical']:
            if not isEnglish(turn_null_to_str(e['label']) + turn_null_to_str(e['text']) + turn_null_to_str(e['text-hint'])):
                continue
            context_info += "There is a component on the same vertical line as this input component. "
            context_info += component_basic_info(e)
    if len(jsondata['same-horizon']) > 0 or len(jsondata['same-vertical']) > 0:
        text_context_info = text_context_info.replace('<context information>', context_info)
    else:
        text_context_info = ""
    if jsondata['id'] in (None, ''):
        text_id = ""
    else:
        EditText_id = jsondata['id'].split('/')[-1].replace('_', ' ')
        text_id = text_id.replace('<EditText id>', EditText_id)
    question = text_header + text_app_name + text_activity_name + text_label + text_text + text_context_info + text_id + text_ask
    final_text = question
    return final_text

def getOutput(question: str, max_new_tokens: int = 50):
    inputs = tokenizer(question, return_tensors='pt', max_length=512, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    max_length = len(input_ids[0]) + max_new_tokens
    if max_length > 1024:
        max_length = 1024

    generated_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        max_new_tokens=100
    )

    return tokenizer.decode(generated_outputs[0], skip_special_tokens=True)

import os
import time

def show_hint(bounds: list, hint_text: str):
    try:
        if len(bounds) >= 4:
            x1, y1, x2, y2 = map(int, bounds)
            
            h = y2 - y1
            if h < 100:
                y2 = y1 + 100
            
            # Encode hint_text for shell command safety
            hint_text_encoded = hint_text.replace('"', '\\"')  # Escape double quotes
            
            # Construct the ADB shell input command to simulate tapping on the input area
            cmd_tap = f"adb shell input tap {x1} {y1}"
            os.system(cmd_tap)
            
            # Introduce a short delay (adjust as needed)
            time.sleep(0.5)
            
            # Construct the ADB shell input command to simulate typing the hint_text
            cmd_text = f"adb shell input text \"{hint_text_encoded}\""
            os.system(cmd_text)
            
            print(f"Suggested hint '{hint_text}' displayed at bounds {bounds}")
        else:
            print(f"Bounds information not complete or invalid: {bounds}")
    except Exception as e:
        print(f"Error showing hint: {e}")

def generate_and_show_hints(correct_hints):
    try:
        for index, (e_component, hint_text) in enumerate(correct_hints):
            bounds = e_component['@bounds']
            res = parse_bounds(bounds)
            show_hint(res, hint_text)
            print(f"Hint {index + 1} generated and displayed: '{hint_text}'")
    except Exception as e:
        print(f"Error generating or showing hints: {e}")
def get_user_feedback(hint_index: int) -> bool:
    response = input(f"Was hint {hint_index} correct? (yes/no): ").strip().lower()
    return response == 'yes'

def find_error_message_below_edittext(edittext_component, all_components):
    edittext_bounds = parse_bounds(edittext_component['@bounds'])
    edittext_bottom = edittext_bounds[3]

    for component in all_components:
        component_bounds = parse_bounds(component['@bounds'])
        component_top = component_bounds[1]

        if (component_top == edittext_bottom and 
            component_bounds[0] <= edittext_bounds[2] and 
            component_bounds[2] >= edittext_bounds[0] and 
            component['@class'] == 'android.view.View' and 
            component.get('@content-desc')):
            return component
    return None
def apply_feedback(feedback_log):
    positive_feedback = sum(1 for feedback in feedback_log if feedback['feedback'] == 'yes')
    negative_feedback = sum(1 for feedback in feedback_log if feedback['feedback'] == 'no')
    total_feedback = len(feedback_log)

    if total_feedback == 0:
        return 0

    reward = positive_feedback - negative_feedback
    print(f"Total feedback: {total_feedback}, Positive: {positive_feedback}, Negative: {negative_feedback}, Reward: {reward}")

    return reward
feedback_log = []
def count_feedback_entries(filename='feedback_log.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return len(data)
    else:
        return 0

# Example function to save feedback log to JSON file
def save_feedback_log(feedback_log, filename='/Users/sanvishukla/Desktop/SRIP/HintDroid-main/HintDroid/feedback_log.json'):
    # Create a list to store formatted feedback entries
    formatted_feedback = []
    
    # Iterate through each entry in feedback_log
    for entry in feedback_log:
        # Extract components from entry
        component = entry['component']
        hint = entry['hint']
        feedback = entry['feedback']
        
        # Construct a formatted feedback entry
        formatted_entry = {
            'prompt': entry.get('prompt', ''),  # Example: Add prompt if available
            'hint': hint,
            'feedback': feedback,
            'reward': entry.get('reward', 0.0)  # Example: Add reward if available
        }
        
        # Append formatted entry to formatted_feedback list
        formatted_feedback.append(formatted_entry)
    
    # Save formatted_feedback to JSON file
    with open(filename, 'w') as f:
        json.dump(formatted_feedback, f, indent=4)

# Function to trigger model fine-tuning
def trigger_model_finetune():

    os.system('python modeltrain2.py')

while True:
    print('Connect to device...')
    d = u2.connect()
    print('Device connected.')
    print(d.info)
    
    save_path = "/Users/sanvishukla/Desktop/SRIP/HintDroid-main/HintDroid/"
    
    def update_hierarchy():
        page_source = d.dump_hierarchy(compressed=True, pretty=True)
        with open(save_path + 'hierarchy.xml', 'w', encoding='utf-8') as xml_file:
            xml_file.write(page_source)
    
    def load_hierarchy():
        with open(save_path + 'hierarchy.xml', 'r', encoding='utf-8') as xml_file:
            return xmltodict.parse(xml_file.read())
    
    update_hierarchy()
    print('Reading hierarchy tree...')
    data_dict = load_hierarchy()
    
    all_components = getAllComponents(data_dict)
    print('All components nums:' + str(len(all_components)))
    
    components_with_edit_text = find_EditText(data_dict)
    print('EditText components nums:' + str(len(components_with_edit_text)))
    
    no_hint_text = [e for e in components_with_edit_text if e['@content-desc'] == '']
    print('EditText with no hint nums:' + str(len(no_hint_text)))
    
    correct_hints = []
    for e_component in no_hint_text:
        print('---------------')
        pprint.pprint(e_component)
        print('---------------')
        bounds = e_component['@bounds']
        dict_info = get_basic_info(e_component)
        same_horizon_components, same_vertical_components = chooseFromPos(all_components, bounds)
        dict_info['same-horizon'] = [get_basic_info(e_hor_component) for e_hor_component in same_horizon_components]
        dict_info['same-vertical'] = [get_basic_info(e_ver_component) for e_ver_component in same_vertical_components]
        dict_info['activity_name'] = ''
        pprint.pprint(dict_info)
        final_text = use_context_info_generate_prompt(dict_info)
        print(final_text)
        try:
            output = getOutput(final_text, max_new_tokens=50)
            split_output = output.split("'")
            if len(split_output) > 1:
                real_ans = split_output[1]
                print(f"We think you should use ({real_ans}) as hint text.")
                correct_hints.append((e_component, real_ans))
            else:
                print("Split operation did not yield expected results.")
        except Exception as e:
            print(f"Error generating or processing output: {e}")
    
    print("\nFinished generating hints.")
    print("Now setting the hints as placeholders on the input fields and printing them.")
    
    generate_and_show_hints(correct_hints)
    
    # Ask for feedback on generated hints
    for hint_index, (e_component, real_ans) in enumerate(correct_hints):
        user_input = input(f"\nWas the hint ({real_ans}) correct for this EditText component? (yes/no): ").strip().lower()
        while user_input not in ['yes', 'no']:
            print("Invalid input. Please enter 'yes' or 'no'.")
            user_input = input(f"Was the hint ({real_ans}) correct for this EditText component? (yes/no): ").strip().lower()
        
        feedback_log.append({
            'component': e_component,
            'hint': real_ans,
            'feedback': user_input
        })
        
        if user_input == 'no':
            print("Hint was not correct. Re-fetching hierarchy to find error message...")
            update_hierarchy()
            data_dict = load_hierarchy()
            all_components = getAllComponents(data_dict)
            error_message_component = find_error_message_below_edittext(e_component, all_components)
            if error_message_component:
                print("Found error message component:")
                pprint.pprint(error_message_component)
                new_hint = error_message_component.get('@content-desc', 'No hint available')
                print(f"New hint based on error message: {new_hint}")
                bounds = e_component['@bounds']
                res = parse_bounds(bounds)
                show_hint(res, new_hint)
            else:
                print("No error message component found below the EditText.")
    
    # Apply feedback and update model
    reward = apply_feedback(feedback_log)
    
    # Clear feedback log for the next iteration
    feedback_log.clear()
    
    # Save feedback log to JSON
    save_feedback_log(feedback_log, 'feedback_log.json')
    
    # Check if feedback log reaches 100 entries to trigger model fine-tuning
    if count_feedback_entries('feedback_log.json') >= 100:
        print("Triggering model fine-tuning...")
        trigger_model_finetune()
    
    time.sleep(15)
