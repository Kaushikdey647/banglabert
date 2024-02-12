import sys
import os
import json
from tqdm import tqdm
import string
from collections import Counter

def main():
    # Check if the correct number of command-line arguments are provided

    print("arguments: ", sys.argv)
    if len(sys.argv) == 1:
        input_folder = "sample_inputs"
        output_folder = "outputs"
    elif len(sys.argv) != 3:
        print("Usage: python compile_result.py <input_folder> <output_folder>")
        return
    else:
        # Get the input and output folder paths from the command-line arguments
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]

    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print("Input folder does not exist.")
        return

    # Check if the output folder exists
    if not os.path.exists(input_folder):
        print("Output folder does not exist.")
        return
    
    test_json = input_process(input_folder)
    pred_json, n_best_pred_json, combined_data = output_process(output_folder)

    if combined_data is None:
        em_list = []
        f1_list = []
        para_num_list = []
        for i,entry in tqdm(enumerate(test_json['data'])):
            title_em_list = []
            title_f1_list = []
            q_num_list = []
            for j,paragraph in enumerate(entry['paragraphs']):
                para_em_list = []
                para_f1_list = []
                for k,qas in enumerate(paragraph['qas']):
                    #append the predictions to the qas
                    test_json['data'][i]['paragraphs'][j]['qas'][k]['best_prediction'] = pred_json[qas['id']]
                    test_json['data'][i]['paragraphs'][j]['qas'][k]['n_best_predictions'] = n_best_pred_json[qas['id']]
                    # Calculate Exact Match and F1
                    try:
                        act = test_json['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text']
                        pred = pred_json[qas['id']]
                        em = calc_em(normalize_str(act),normalize_str(pred))
                        f1 = calc_f1(normalize_str(act),normalize_str(pred))
                    except:
                        em,f1 = 0,0
                    
                    #add entries to json
                    test_json['data'][i]['paragraphs'][j]['qas'][k]['exact_match'] = em
                    test_json['data'][i]['paragraphs'][j]['qas'][k]['f1_score'] = f1

                    # append values to paragraph lists
                    para_em_list.append(em)
                    para_f1_list.append(f1)

                para_q_num = len(para_em_list)
                para_em_count = sum(para_em_list)
                average_f1 = sum(para_f1_list) / len(para_f1_list)
                test_json['data'][i]['paragraphs'][j]['q_num'] = para_q_num
                test_json['data'][i]['paragraphs'][j]['em_count'] = para_em_count
                test_json['data'][i]['paragraphs'][j]['average_f1'] = average_f1
                q_num_list.append(para_q_num)
                title_em_list.append(para_em_count)
                title_f1_list.append(average_f1)
            title_q_num = sum(q_num_list)
            title_em_count = sum(title_em_list)
            title_av_f1 = sum(title_f1_list)/len(title_f1_list)
            test_json['data'][i]['q_num'] = title_q_num
            test_json['data'][i]['em_count'] = title_em_count
            test_json['data'][i]['average_f1'] = title_av_f1
            para_num_list.append(title_q_num)
            em_list.append(title_em_count)
            f1_list.append(title_av_f1)
        total_q_num = sum(para_num_list)
        total_em_count = sum(em_list)
        total_av_f1 = sum(f1_list)/len(f1_list)
        test_json['metrics'] = {}
        test_json['metrics']['total_q_num'] = total_q_num
        test_json['metrics']['total_em_count'] = total_em_count
        test_json['metrics']['total_average_f1'] = total_av_f1
            
        combined_data = test_json
        #save test_json to output folder
        with open(os.path.join(output_folder, "combined_data.json"), "w", encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False)
    
    
def input_process(input_folder):
    # Check if the input folder contains the necessary files
    if not os.path.exists(os.path.join(input_folder, "test.json")):
        print("ERROR: test.json file does not exist in the input folder.")
        return

    # Read the contents of the JSON files
    with open(os.path.join(input_folder, "test.json"), "r") as f:
        test_json = json.load(f)

    return test_json
    

def output_process(output_folder):
    # Check if the output folder contains the necessary files
    if not os.path.exists(os.path.join(output_folder, "eval_nbest_predictions.json")):
        print("ERROR: eval_nbest_predictions.json file does not exist in the output folder.")
        return

    if not os.path.exists(os.path.join(output_folder, "eval_predictions.json")):
        print("ERROR: eval_predictions.json file does not exist in the output folder.")
        return
    

    # Read the contents of the JSON files
    with open(os.path.join(output_folder, "eval_nbest_predictions.json"), "r") as f:
        n_best_pred_json = json.load(f)

    with open(os.path.join(output_folder, "eval_predictions.json"), "r") as f:
        pred_json = json.load(f)

    
    if not os.path.exists(os.path.join(output_folder, "combined_test.json")):
        print("WARNING: combined_test.json file does not exist in the output folder.")
        combined_data = None
    else:
        with open(os.path.join(output_folder, "combined_test.json"), "r") as f:
            combined_data = json.load(f)

    return pred_json, n_best_pred_json, combined_data
    
def normalize_str(inp):
    # remove bengali punctuations
    inp = ''.join(
        ch for ch in inp if ch not in set(string.punctuation) and ch not in ['।', '॥', '-']
    )
    
    # remove whitespaces
    inp = ' '.join(
        inp.strip().split()
    )
    return inp

def calc_em(pred, act):
    if pred == act:
        return 1
    else:
        return 0

def calc_f1(pred, act):
    pred_tok = pred.split()
    act_tok = act.split()
    comm_tok = Counter(act_tok) & Counter(pred_tok)
    num_same = sum(comm_tok.values())
    if num_same == 0:
        return 0
    precision = 1.0*num_same/len(pred_tok)
    recall = 1.0*num_same/len(act_tok)
    f1 = (2*precision*recall)/(precision+recall)
    return f1

if __name__ == "__main__":
    main()
