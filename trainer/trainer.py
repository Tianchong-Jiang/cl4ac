import torch
from tqdm import tqdm
import numpy as np
import random
from warmup_scheduler import GradualWarmupScheduler
from datetime import datetime
# from bert_tools.custom_tokenizer import CUSTOM_TOKENIZER
from data_loader.clotho_dataset import ClothoDataset, get_dataloader
from evaluation.eval_model import eval_model
from model.TransModel import TransformerModel
from transformers import AutoTokenizer
from trainer.get_gradient import get_total_grad_norm
from trainer.logger import Logger
from trainer.loss import get_loss, calculating_weight
from trainer.optimizer import get_optimizer
from utils.save_load_model import save_model
from w2v_tools.w2v_model import Word2Vec
import os
import csv
import wandb
import pdb


def train(config, device):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d+%H-%M-%S")
    setup_seed(config.training.seed)
    if config.w2v.enable:
        tokenizer = Word2Vec(w2v_model_path=config.w2v.w2v_path, multisos=config.multisos.enable)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.bert.bert_path)
    train_dataset = ClothoDataset('/data', config,
                                  tokenizer=tokenizer)
    test_dataset = ClothoDataset('/data', config,
                                 tokenizer=tokenizer, is_train=False)
    train_loader = get_dataloader(train_dataset, config, tokenizer, is_train=True, multisos=config.multisos.enable)
    test_loader = get_dataloader(test_dataset, config, tokenizer, is_train=False, multisos=config.multisos.enable)
    model = TransformerModel(config).to(device)
    auxilary_criteria = get_loss(config)
    criteria = get_loss(config)
    optimizer = get_optimizer(config, model)
    if config.training.activate_weight_on_loss.enable:
        addition_weight_ratio = train_dataset.get_word_frequency(tokenizer, config)
        weight = calculating_weight(tokenizer, addition_weight_ratio,
                                    reduce_punc=config.training.activate_weight_on_loss.reduce_punc_weight,
                                    reduce_stopwords=config.training.activate_weight_on_loss.reduce_punc_weight)
        criteria.weight = torch.tensor(weight).to(device)
    # Must be set after weight calculation, since [PAD] weight will be set as 0
    if config.bert.use_sep_as_pad:
        tokenizer.pad_token = tokenizer.sep_token
    logger = Logger(config)
    steps = 0
    scheduler_warmup = None
    if config.optimizer.warm_up.enable:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    for epoch in range(config.training.epoch):
        epoch_loss = []
        pbar = tqdm(train_loader, total=train_loader.__len__(),
                    position=0, leave=True, ascii=True, desc="Epoch {}".format(epoch))
        if config.optimizer.warm_up.enable:
            scheduler_warmup.step(epoch + 1)

        for data in pbar:
            model.train()
            optimizer.zero_grad()
            attention_mask = None if 'attention_mask' not in data.keys() else data['attention_mask'].to(device)
            max_non_pad_indexes = None
            max_neg_non_pad_indexes = None
            if config.auxiliary_task.use_last_hidden:
                non_pad_indexes = (data['inputs'] != tokenizer.pad_token_id).nonzero()
                max_non_pad_indexes = torch.zeros((data['inputs'].shape[0]))
                for non_pad_index in non_pad_indexes.detach().cpu().numpy().tolist():
                    max_non_pad_indexes[non_pad_index[0]] = non_pad_index[1]
                neg_non_pad_indexes = (data['negative_inputs'] != tokenizer.pad_token_id).nonzero()
                max_neg_non_pad_indexes = torch.zeros((data['negative_inputs'].shape[0]))

                for non_pad_index in neg_non_pad_indexes.detach().cpu().numpy().tolist():
                    max_neg_non_pad_indexes[non_pad_index[0]] = non_pad_index[1]

            if max_non_pad_indexes is not None:
                max_non_pad_indexes=max_non_pad_indexes.to(device)

            if max_neg_non_pad_indexes is not None:
                max_neg_non_pad_indexes=max_neg_non_pad_indexes.to(device)

            y_hat = model(data['audio_embedding'].to(device), data['inputs'].to(device),
                          attention_mask=attention_mask,
                          selection_result=config.auxiliary_task.selection_loss,
                          max_non_pad_indexes=max_non_pad_indexes)
            if config.auxiliary_task.selection_loss:
                y_hat, selection_score = y_hat
                _, negative_selection_score = model(data['audio_embedding'].to(device),
                                                    data['negative_inputs'].to(device),
                                                    attention_mask=attention_mask,
                                                    selection_result=config.auxiliary_task.selection_loss,
                                                    max_non_pad_indexes=max_neg_non_pad_indexes)
                selection_labels = [1] * selection_score.shape[0] + [0] * negative_selection_score.shape[0]
                pos_neg_selection_scores = torch.cat([selection_score, negative_selection_score])
                selection_loss = auxilary_criteria(pos_neg_selection_scores,
                                          torch.tensor(selection_labels).to(device).contiguous().view(-1))
                selection_loss = selection_loss * config.auxiliary_task.weight_factor
            if config.bert.auto_regression and config.bert.auto_regressive_gamma < 1.0:
                losses = None
                for batch_index in range(y_hat.shape[0]):
                    y_hat_iter = y_hat[batch_index]
                    input_iter = data['inputs'][batch_index]
                    targets_iter = data['targets'][batch_index]
                    known_length = torch.nonzero(input_iter != tokenizer.mask_token_id).shape[0]
                    gamma = config.bert.auto_regressive_gamma ** known_length
                    loss_iter = criteria(y_hat_iter, targets_iter.to(device)) * gamma
                    if losses is None:
                        losses = loss_iter
                    else:
                        losses += losses
                losses /= y_hat.shape[0]
                loss = losses
            else:
                loss = criteria(y_hat.contiguous().view(-1, y_hat.shape[-1]),
                                data['targets'].to(device).contiguous().view(-1))
            wandb.log({"pred_loss": loss, "step": steps})
            wandb.log({"selection_los": selection_loss, "step": steps})

            if config.auxiliary_task.selection_loss:
                loss += selection_loss
            loss.backward()
            epoch_loss.append(loss.detach().cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            gradients = get_total_grad_norm(model.parameters())
            optimizer.step()
            logger.add_train_loss(loss.detach().cpu().item(), steps)
            logger.add_train_grad(gradients, steps)

            # sampling from the top-k tokens
            if config.sampling.topk > 1:
                topk_indices = torch.topk(y_hat, k=config.sampling.topk, dim=-1).indices.detach().cpu().numpy()
                random_indices = np.random.randint(0, config.sampling.topk, size=(topk_indices.shape[1]))
                token_ids = topk_indices[:, np.arange(topk_indices.shape[1]), random_indices][0].tolist()
            else:
                token_ids = torch.argmax(y_hat, dim=-1).detach().cpu().numpy().tolist()[0]

            # show the output text on progress bar
            pbar.set_postfix_str("loss: {:.4f}, gradient: {:.4f} pred: {}".format(
                loss.detach().cpu().item(), gradients,
                tokenizer.decode(token_ids)
            ))
            steps += 1

            # log loss
            wandb.log({"loss": loss, "step": steps})

            # log text output
            if steps % 10 == 0:
                # pdb.set_trace()
                table = wandb.Table(columns=["Step", "Text"])
                table.add_data(str(steps), tokenizer.decode(token_ids))
                wandb.log({"Text output": table})

    # evaluate (inference)
    pbar_test = tqdm(test_loader, total=test_loader.__len__(),
                position=0, leave=True, ascii=True, desc="Epoch {}".format(epoch))
    model.eval()
    output = []
    for epoch in range(config.testing.epoch):
        for test_data in pbar_test:
            attention_mask = None if 'attention_mask' not in test_data.keys() else test_data['attention_mask'].to(device)
            max_non_pad_indexes = None
            max_neg_non_pad_indexes = None
            if config.auxiliary_task.use_last_hidden:
                non_pad_indexes = (test_data['inputs'] != tokenizer.pad_token_id).nonzero()
                max_non_pad_indexes = torch.zeros((test_data['inputs'].shape[0]))
                for non_pad_index in non_pad_indexes.detach().cpu().numpy().tolist():
                    max_non_pad_indexes[non_pad_index[0]] = non_pad_index[1]
                neg_non_pad_indexes = (test_data['negative_inputs'] != tokenizer.pad_token_id).nonzero()
                max_neg_non_pad_indexes = torch.zeros((test_data['negative_inputs'].shape[0]))

                for non_pad_index in neg_non_pad_indexes.detach().cpu().numpy().tolist():
                    max_neg_non_pad_indexes[non_pad_index[0]] = non_pad_index[1]

            if max_non_pad_indexes is not None:
                max_non_pad_indexes=max_non_pad_indexes.to(device)

            if max_neg_non_pad_indexes is not None:
                max_neg_non_pad_indexes=max_neg_non_pad_indexes.to(device)

            y_hat, _ = model(test_data['audio_embedding'].to(device), test_data['inputs'].to(device),
                            attention_mask=attention_mask,
                            selection_result=config.auxiliary_task.selection_loss,
                            max_non_pad_indexes=max_non_pad_indexes)

            if config.sampling.topk > 1:
                topk_indices = torch.topk(y_hat, k=config.sampling.topk, dim=-1).indices.detach().cpu().numpy()
                random_indices = np.random.randint(0, config.sampling.topk, size=(topk_indices.shape[1]))
                token_ids = topk_indices[:, np.arange(topk_indices.shape[1]), random_indices].tolist()
            else:
                token_ids = torch.argmax(y_hat, dim=-1).detach().cpu().numpy().tolist()

            # decode the token ids
            for i in range(y_hat.shape[0]):
                text = tokenizer.decode(token_ids[i])
                output.append([test_data['filename'][i], epoch, text])

            pbar_test.set_postfix_str("test time, pred: {}".format(
                tokenizer.decode(token_ids[0])
            ))

    with open("/rmx/diffuser/output/output_10.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output)

    # Save model after training is done
    # model_path = f'/rmx/diffuser/output/{config.experiment.name}-{current_time}/'
    # os.makedirs(model_path, exist_ok=True)
    # model_path += "{:04d}.pt".format(epoch)
    # save_model(model_path, model, optimizer, epoch, config, np.asarray(epoch_loss).mean())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
