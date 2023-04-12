import json
import argparse
from controlled_text_generation import *
import os

def get_label_text(input_path):
    input_data=[]
    text_data=[]
    with open(input_path, 'r') as ff:
        for row in ff:
            word_lst = row.split('||')[1]
            label_mapping=split_label(row.split('||')[0])
            input_data.append(label_mapping)
            text_data.append(word_lst)
    
    return input_data,text_data

def get_classifier_logits(classifer_model,classifier_tokenizer,text,id2label,label2id):
  encoding = classifier_tokenizer(text, return_tensors="pt")
  encoding = {k: v.to(classifer_model.device) for k,v in encoding.items()}

  outputs = classifer_model(**encoding) 
  
  logits = outputs.logits
  # apply sigmoid + threshold
#   sigmoid = torch.nn.Sigmoid()
#   probs = sigmoid(logits.squeeze().cpu())
#   predictions = np.zeros(probs.shape)
#   predictions[np.where(probs >= 0.5)] = 1
#   # turn predicted id's into actual label names
#   predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
  # print(predicted_labels)

  return logits

def get_classifier_report(class_checked,label_data_train,text_data_train,model,tokenizer,id2label,label2id):
    total=0
    correct=0
    incorrect=[]
    predictions=[]
    for labels,text in zip(label_data_train,text_data_train):
        # print("text: ",text)
        # print("label : ",labels)
        if class_checked not in labels :
            continue
        logits=get_classifier_logits(model,tokenizer,text,id2label,label2id)
        # print("classifier : ",logits)
        pred=id2label[np.argmax(logits.detach().numpy()[0])]
        # print("prediction : ",pred)
        total+=1
        predictions.append(pred)
        if pred==labels[class_checked]:
            correct+=1
        else :
            incorrect.append((text,labels))

    accuracy_score=correct/total
    print("Correct = {}/{}".format(correct,total))
    # print("Incorrect : ")
    # for i in incorrect:
    #     print(i[1])
    #     print(i[0])
    
    return accuracy_score,predictions,incorrect

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='classifier report args')
    parser.add_argument('--classifier_folder', type=str, default='', help='')
    parser.add_argument('--class_type', type=str, default='area', help='')
    parser.add_argument('--dataset_path', type=str, default='dataset/e2e_data/', help='')
    # COMMAND : python3 classifier_report.py --classifier_folder classifier_models/models_for_area --class_type area --dataset_path dataset/e2e_data

    args = parser.parse_args()

    MODELS_FOLDER=args.classifier_folder
    CLASS_LABEL=args.class_type
    DATASET_PATH=args.dataset_path

    control_list=[CLASS_LABEL]
    label2id_dict,id2label_dict=read_all_label_maps(control_list)
    print(label2id_dict.keys())
    label2id=label2id_dict[CLASS_LABEL]
    id2label=id2label_dict[CLASS_LABEL]
    n_labels=len(id2label)

    if CLASS_LABEL=='family_friendly':
        CLASS_LABEL='family friendly'
    elif CLASS_LABEL=='customer_rating':
        CLASS_LABEL='customer rating'

    input_path_train=os.path.join(DATASET_PATH,'src1_train.txt')
    label_data_train,text_data_train=get_label_text(input_path_train)

    input_path_test=os.path.join(DATASET_PATH,'src1_test.txt')
    label_data_test,text_data_test=get_label_text(input_path_test)

    input_path_valid=os.path.join(DATASET_PATH,'src1_valid.txt')
    label_data_valid,text_data_valid=get_label_text(input_path_valid)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    checkpoint_paths=os.listdir(MODELS_FOLDER)
    report={}
    for checkpoint_path in checkpoint_paths:
        if 'checkpoint' not in checkpoint_path:
            continue

        checkpoint_report={}
        
        model_path=os.path.join(MODELS_FOLDER,checkpoint_path)
        print("Generating report for {}".format(model_path))
        model,tokenizer=get_classifier(model_path,n_labels,device=device)

        print("Test report ")
        accuracy_score,predicitions,incorrect=get_classifier_report(CLASS_LABEL,label_data_test,text_data_test,model,tokenizer,id2label,label2id)
        checkpoint_report['test']={'accuracy':accuracy_score,'predictions':predicitions,'incorrect':incorrect}

        print("Valid report ")
        accuracy_score,predicitions,incorrect=get_classifier_report(CLASS_LABEL,label_data_valid,text_data_valid,model,tokenizer,id2label,label2id)
        checkpoint_report['valid']={'accuracy':accuracy_score,'predictions':predicitions,'incorrect':incorrect}

        print("Train report ")
        accuracy_score,predicitions,incorrect=get_classifier_report(CLASS_LABEL,label_data_train,text_data_train,model,tokenizer,id2label,label2id)
        checkpoint_report['train']={'accuracy':accuracy_score,'predictions':predicitions,'incorrect':incorrect}

        
        report[model_path]=checkpoint_report


    output_file=os.path.join(MODELS_FOLDER,'report.json')
    
    with open(output_file, 'w') as fp:
        json.dump(report, fp)
        
    print("Report written to ",output_file)
