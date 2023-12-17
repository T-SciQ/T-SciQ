import os
import re
import sys
import math
import json
import argparse
import random
import time
import torch
import openai

import numpy as np
import torch.nn.functional as F

import utils

from base_prompt import *
from model import *

sys.path.append("../")
from trainer import MMCoTTrainer
from transformers import T5Tokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from model_mmcot import T5ForMultimodalGeneration
from utils_data import img_shape,  ScienceQADatasetImg


openai.api_key = os.getenv("OPENAI_API_KEY")


def load_data(args):
    problems_train = json.load(open(os.path.join(args.data_root, "problems_train.json")))
    problems_test_qar = json.load(open(os.path.join(args.data_root, "problems_test_qar.json")))
    problems_test_ps = json.load(open(os.path.join(args.data_root, "problems_test_ps.json")))

    train_pids = list(problems_train.keys())
    test_pids = list(problems_test_qar.keys())

    return problems_train, problems_test_qar, problems_test_ps, train_pids, test_pids


def get_output(test_set, tokenizer, args):

    def compute_metrics_acc(eval_preds):
        preds = eval_preds.predictions[0]
        targets = eval_preds.label_ids
        # preds = preds.argmax(axis=2)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct = 0
        assert len(preds) == len(targets)
        for idx, pred in enumerate(preds):
            reference = targets[idx]
            reference = extract_ans(reference)
            extract_pred = extract_ans(pred)
            best_option = extract_pred
            if reference == best_option:
                correct +=1 
        return {'accuracy': 1.0*correct/len(targets)}

    patch_size = img_shape["detr"]
    padding_idx = tokenizer._convert_token_to_id(tokenizer.pad_token)
    save_dir = "/data1/huyi/codes/mm-cot/experiments/answer_-data1-huyi-codes-mm-cot-unifiedqa-t5-base_policy"
    model = T5ForMultimodalGeneration.from_pretrained("/data1/huyi/codes/mm-cot/models/MM-CoT-UnifiedQA-base-Answer", patch_size=patch_size, padding_idx=padding_idx, save_dir=save_dir) 

    datacollator = DataCollatorForSeq2Seq(tokenizer)
    print("model parameters: ", model.num_parameters())

    training_args = Seq2SeqTrainingArguments(
        save_dir,
        do_train=False,
        do_eval=False,
        evaluation_strategy="no",
        logging_strategy="steps",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=5e-5,
        eval_accumulation_steps=None,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        num_train_epochs=20,
        predict_with_generate=None,
        report_to="none",
    )

    trainer = MMCoTTrainer(
        model=model,
        args=training_args,
        train_dataset=test_set,
        eval_dataset=test_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_acc
    )

    predict_results = trainer.predict(test_dataset=test_set, max_length=64)
    pred = predict_results.predictions[0]
    target = predict_results.label_ids
    pred = tokenizer.batch_decode(
        pred, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    target = tokenizer.batch_decode(
        target, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    extract_pred = extract_ans(pred)
    if extract_pred != "FAILED":
        return extract_pred
    else:
        return random.choice(args.options) # random choose one option


def extract_ans(ans):
    pattern = re.compile(r'The answer is \(([A-Z])\)')
    res = pattern.findall(ans)
    
    if len(res) == 1:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED" 
    return answer 


def get_batch_reward_loss(scores, problems_test_qar, problems_test_ps, test_pid_batch, label_batch, args):
    batch_loss = 0
    batch_reward = 0
    
    tokenizer = T5Tokenizer.from_pretrained("/data1/huyi/codes/mm-cot/unifiedqa-t5-base")
    name_maps = json.load(open('/data1/huyi/codes/mm-cot/vision_features/name_map.json'))
    image_features = np.load('/data1/huyi/codes/mm-cot/vision_features/detr.npy')
    ## loop over the training examples
    for i in range(len(scores)):

        # interact with the environment to get rewards, which in our case is to feed the prompt into GPT-3 and evaluate the prediction
        cand_prob = scores[i, :].clone().detach()
        cand_prob = cand_prob.cpu().numpy()
        cand_prob = np.nan_to_num(cand_prob, nan=0.000001)  # replace np.nan with 0
        cand_prob /= cand_prob.sum()  # make probabilities sum to 1
        # print(f"cand_prob: {cand_prob}")

        # sample shot_pids from the cand_prob distribution
        cid = np.random.choice(range(len(cand_prob)), args.shot_number, p=cand_prob, replace=False)

        # get the output from MM-CoT
        if cid.item() == 0:
            test_set = ScienceQADatasetImg(
                problems_test_qar,
                [test_pid_batch[i]],
                name_maps,
                tokenizer,
                512,
                64,
                args,
                image_features,
            )
        else:
            test_set = ScienceQADatasetImg(
                problems_test_ps,
                [test_pid_batch[i]],
                name_maps,
                tokenizer,
                512,
                64,
                args,
                image_features,
            )
        prediction = get_output(test_set, tokenizer, args)

        log_prob = torch.log(scores[i, cid.item()])

        if prediction.lower() == label_batch[i].lower():
            _reward = 1
        else:
            _reward = -1

        batch_reward += _reward
        batch_loss -= _reward * log_prob

    return cid, batch_reward, batch_loss


def policy_gradient_train(policy_model, problems_train, problems_test_qar, problems_test_ps, train_pids, test_pids, args):
    # REINFORCE
    # if os.path.exists(args.ckpt_path):
    #     print("!!! Model dir already exists. Consider load it instead of training again.")

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr)

    train_samples = []
    for pid in train_pids:
        train_samples.append(create_train_prompt(pid, problems_train))
    test_qar_samples, test_ps_samples = [], []
    labels = []
    for pid in test_pids:
        test_qar_samples.append(create_test_prompt(pid, problems_test_qar))
        test_ps_samples.append(create_test_prompt(pid, problems_test_ps))
        answer = get_answer(problems_test_qar[pid], args.options)
        labels.append(answer)

    num_batch = math.ceil(len(train_samples) / args.batch_size)

    reward_history = []
    loss_history = []

    total_reward_history = []  # epoch based
    total_loss_history = []  # epoch based

    STOP_FLAG = False

    for epoch in range(args.epochs):
        logger.write(f"Epoch: {epoch}")

        total_train_reward = 0
        total_train_loss = 0

        # We can simply set the batch_size to len(train_data) in few-shot setting.
        for batch_i in range(num_batch):
            logger.write(f"Batch: {batch_i}")
            train_batch = train_samples[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            label_batch = labels[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            test_pid_batch = test_pids[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            # unit_batch = units[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            # option_batch = options[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            embedding_test = []
            test_examples = []
            for index in range(args.batch_size):
                qar_example = test_qar_samples[batch_i * args.batch_size + index]
                ps_example = test_ps_samples[batch_i * args.batch_size + index]
                test_example = [qar_example, ps_example]
                test_examples.append(test_example)
                embedding_test.append(policy_model(test_example))  # [(2, embedding_size), ...]
            
            embedding_train = policy_model(train_batch)  # len(train_batch) x embedding_size

            scores_list = []
            for index in range(args.batch_size):
                scores_list.append(torch.mm(embedding_train[index].unsqueeze(0), embedding_test[index].t())) # (1, 2)
            scores = torch.concat(scores_list, dim=0) # len(train_batch) x 2

            scores = F.softmax(scores, dim=1)  # len(train_batch) x 2

            cid, reward, loss = get_batch_reward_loss(scores, problems_test_qar, problems_test_ps, test_pid_batch, label_batch, args)

            logger.write(f"cids for sample[-1] in batch: {cid.item()}")
            logger.write(f"Cand prob for sample[-1] in batch: {[round(x,5) for x in scores[-1, :].tolist()]}")
            logger.write(f"### reward for the batch: {reward}")
            logger.write(f"### loss for the batch: {loss}\n")

            # linear layer has Weight and bias
            # prev_param = list(policy_model.linear.parameters())[0].clone()
            # print(f"prev_param: {prev_param.data}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for each iteration/batch
            total_train_reward += reward
            total_train_loss += loss.item()

            reward_history.append(reward)
            loss_history.append(loss.item())

            if np.isnan(loss.item()):
                STOP_FLAG = True
                break

        # for each epoch
        total_reward_history.append(total_train_reward)
        total_loss_history.append(total_train_loss)

        best_reward = max(total_reward_history)
        best_loss = min(total_loss_history)

        best_reward_epoch = total_reward_history.index(best_reward)
        best_loss_epoch = total_loss_history.index(best_loss)

        logger.write("============================================")
        logger.write(f"### Epoch: {epoch} / {args.epochs}")
        logger.write(f"### Total reward: {total_train_reward}, " + f"Total loss: {round(total_train_loss,5)}, " +
                     f"Best reward: {best_reward} at epoch {best_reward_epoch}, " +
                     f"Best loss: {round(best_loss, 5)} at epoch {best_loss_epoch}\n")

        # save every epoch
        ckpt_file = os.path.join(args.ckpt_path, f"ckpt_{epoch}.pt")
        torch.save(policy_model.linear.state_dict(), ckpt_file)
        logger.write(f"saved the ckpt to {ckpt_file}")

        # save best epoch
        if epoch == best_reward_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_reward.pt")
            torch.save(policy_model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best reward ckpt to {ckpt_file}")

        if epoch == best_loss_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_loss.pt")
            torch.save(policy_model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best loss ckpt to {ckpt_file}")

        # save reward and loss history
        history = {
            "reward_history": reward_history,
            "loss_history": loss_history,
            "total_reward_history": total_reward_history,
            "total_loss_history": total_loss_history,
        }
        history_file = os.path.join(args.ckpt_path, "history.json")
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, separators=(',', ': '))

        # print cache info
        logger.write("============================================\n")

        if STOP_FLAG:
            break

    # save in the end
    ckpt_file = os.path.join(args.ckpt_path, "ckpt_final.pt")
    torch.save(policy_model.linear.state_dict(), ckpt_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data1/huyi/codes/mm-cot/policy/data')
    parser.add_argument('--model', type=str, default='gpt3_rl')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--prompt_format', type=str, default='QCMG-A', help='prompt format template',
                        choices=['QCM-A', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE'])

    # User options
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--shot_number', type=int, default=1, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='text-davinci-002', choices=['text-davinci-002', 'ada'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    # Policy gradient settings
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_config',
                        type=str,
                        default='bert-base-uncased',
                        choices=['distilbert-base-uncased', 'bert-base-uncased'])
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of policy network.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--embedding_size', type=int, default=128, help='Policy network final layer hidden state size.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=20,
                        help='Policy network training batch size. Set to train_number by default.')
    parser.add_argument('--ckpt_root', type=str, default='/data1/huyi/codes/mm-cot/policy/checkpoints')

    args = parser.parse_args()

    # print and save the args
    args.ckpt_path = os.path.join(args.ckpt_root, args.label)
    utils.create_dir(args.ckpt_path)
    _logger = utils.Logger(args.ckpt_path + '/args.txt')

    print('====Input Arguments====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args


if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True

    problems_train, problems_test_qar, problems_test_ps, train_pids, test_pids = load_data(args)

    ## policy network
    policy_model = policy_network(model_config=args.model_config,
                                  add_linear=True,
                                  embedding_size=args.embedding_size,
                                  freeze_encoder=True)

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU
    policy_model = policy_model.to(device)

    ## TRAINING
    logger = utils.Logger(os.path.join(args.ckpt_path, 'log.txt'))
    policy_gradient_train(policy_model, problems_train, problems_test_qar, problems_test_ps, train_pids, test_pids, args)
