import torch
import numpy as np
import time
import os
import sys
import ownutils
import model_res
import transformers
import math

import random
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
beta=ownutils.beta

target_device = 0  


def save_checkpoint(model, optimizer, histories=None, append=''):
    if len(append) > 0:
        append = '-' + append
    if not os.path.isdir(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)
    checkpoint_path = os.path.join(opt.checkpoint_path, 'model%s.pth' %(append))
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path), flush=True)
    optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pth' %(append))
    torch.save(optimizer.state_dict(), optimizer_path)
    if histories:
        with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
            ownutils.pickle_dump(histories, f)

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup: int, max_iters: int
    ):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor

def  train(opt):
    batch_size=opt.batch_size
    epoch=1
    iteration=0
    rt_half_range=2

    all_best_score=None
    count_steps=0
    eval_step=opt.eval_step
    # print(mgf_file)
    print(opt.train_file)
    print(opt.val_file)
    ownutils.printParams(opt)

    if opt.spect_file == "" or opt.train_file == "" or opt.val_file == "":
        print("Error: missing some files")
        return


    ms_file_offset=model_res.readmgfOffset(opt.spect_file)#F1:2=[START,END]
    training_dataset=model_res.DIADataset(opt.spect_file,opt.train_file,ms_file_offset,opt,rt_half_range=rt_half_range)
    training_data=DataLoader(training_dataset, collate_fn=model_res.collate_fn,batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
    val_dataset=model_res.DIADataset(opt.spect_file,opt.val_file,ms_file_offset,opt,rt_half_range=rt_half_range)
    val_data=DataLoader(val_dataset, collate_fn=model_res.collate_fn,batch_size=batch_size, num_workers=8, pin_memory=True)

    model=transformers.BiTransformer(512, 8, 1024, 9, 0.1, model_res.seq_size, 10, 9, 2*rt_half_range).cuda()
    model.setBeta(beta)
    base_model = torch.nn.DataParallel(model,device_ids=ownutils.n_gpu)

    optimizer = ownutils.build_optimizer(model.parameters(), opt)
    scheduler=CosineWarmupScheduler(optimizer,warmup=opt.warmup, max_iters=opt.scheduler_maxiters)

    print(f"training start time:{time.ctime()}", flush=True)
    while True:
        start = time.time()
        base_model.train()

        for num_data, data in enumerate(training_data):
            torch.cuda.synchronize()
            optimizer.zero_grad()
            forward_loss = base_model(data, True, True)
            forward_loss.mean().backward()
            optimizer.step()

            optimizer.zero_grad()
            backward_loss = base_model(data, False, True)
            backward_loss.mean().backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            iteration+=1
            count_steps+=1

        
            # # eval model
            if iteration%eval_step==0:
                val_loss = 0
                val_acc = 0
                acc_f = 0
                acc_b = 0
                tmp_ff = 0
                tmp_bb = 0
                max_acc = 0
                total_seq = 0
                base_model.eval()
                with torch.no_grad():
                    for num_data, data in enumerate(val_data):
                        _, len_seq = data[1].shape
                        forward_loss1, tmp_acc_f = base_model(data, True, False)
                        backward_loss1, tmp_acc_b = base_model(data, False, False)
                        val_loss += (forward_loss1.mean()+backward_loss1.mean())*len_seq
                        tmp_ff += forward_loss1.mean()*len_seq
                        tmp_bb += backward_loss1.mean()*len_seq

                        total_seq += len_seq
                        val_acc += tmp_acc_f + tmp_acc_b
                        acc_f += tmp_acc_f
                        acc_b += tmp_acc_b
                        max_acc += max(tmp_acc_b, tmp_acc_f)

                    val_loss = val_loss/(total_seq-1)
                    tmp_ff = tmp_ff/(total_seq-1)
                    tmp_bb = tmp_bb/(total_seq-1)
   
                    val_acc = val_acc/(num_data+1)
                    acc_f = acc_f/(num_data+1)
                    acc_b = acc_b/(num_data+1)
                    max_acc = max_acc/(num_data+1)
                base_model.train()
                end = time.time()
                # tmp_llrr=[param_group["lr"] for param_group in optimizer.param_groups]
                total_train_loss=forward_loss.mean()+backward_loss.mean()
                print(f"epoch {epoch}--{iteration},  train_loss={total_train_loss.item()},val_loss={val_loss}, val_acc={val_acc}, max_acc={max_acc}, time={end-start}", flush=True)
                # print(f"GPU memnory usage: {torch.cuda.max_memory_allocated()/(1024**3)}G")

                current_score = val_acc
                if all_best_score is None or current_score > all_best_score:
                    all_best_score=current_score
                    save_checkpoint(model, optimizer, append=f"final_best")
                    count_steps=0
                if count_steps>700000:
                    all_best_score=float('inf')


        if (epoch >= opt.max_epochs and opt.max_epochs != -1):
            break
        epoch += 1

    
    print(f"training finish time:{time.ctime()}", flush=True)

def generation(opt):
    print(f"start evaluation: {time.ctime()}")
    print(f"model path: {opt.model_path}")
    true_seq=[]
    true_mz=[]
    forward_mz=[]
    rev_mz=[]
    charge=[]
    seq=[]
    aa_score=[]
    pep_score=[]
    rev_seq=[]
    rev_aa_score=[]
    rev_pep_score=[]
    if opt.spect_file == "" or opt.test_file == "":
        print("Error: no spectrum file or no feature file")
        return
    ms_file_offset=model_res.readmgfOffset(opt.spect_file)
    val_dataset=model_res.DIADataset(opt.spect_file,opt.test_file,ms_file_offset,opt,rt_half_range=2)
    val_data=DataLoader(val_dataset, collate_fn=model_res.collate_fn,batch_size=opt.batch_size, num_workers=8, pin_memory=True)


    model=transformers.BiTransformer(512, 8, 1024, 9, 0, model_res.seq_size, 10, 9, 2*2).cuda()

    if opt.model_path=="":
        print("Error: no model path")
        return
    map_location = f'cuda:{target_device}'
    model.load_state_dict(torch.load(opt.model_path,map_location=map_location))
    model.eval()
    print(f"{opt.batch_size}")
    with torch.no_grad():
        for num_data, data in enumerate(val_data):
            cur_seq, cur_rev_seq = model.prediction(data)

            for i in range(len(cur_seq)):

                if data[1][i].cpu().int().tolist() != []:
                    tmp_true,_=model_res.decoderSeq(data[1][i].cpu().int().tolist(),None)
                    true_seq.append(tmp_true)
                else:
                    true_seq.append("")
                true_mz.append(data[4][i][2].item())
                charge.append(data[4][i][1].item())

                if len(cur_seq[i])!=0:
                    try:
                        seq.append(cur_seq[i][0][2])
                        aa_score.append(cur_seq[i][0][1])
                        pep_score.append(cur_seq[i][0][0])
                        forward_mz.append(ownutils.calMass(cur_seq[i][0][2], data[4][i][1].item()))
                    except Exception as e:
                        print("f")
                        print(e)
                        print(data[4][i])
                        print(cur_seq[i])
                        sys.exit(1)
                else:
                    seq.append("")
                    aa_score.append([-1])
                    pep_score.append(-1)
                    forward_mz.append(0.0)

                if len(cur_rev_seq[i])!=0:
                    try:
                        rev_seq.append(cur_rev_seq[i][0][2][::-1])
                        rev_aa_score.append(cur_rev_seq[i][0][1])
                        rev_pep_score.append(cur_rev_seq[i][0][0])
                        rev_mz.append(ownutils.calMass(cur_rev_seq[i][0][2][::-1], data[4][i][1].item()))
                    except Exception as e:
                        print("r")
                        print(e)
                        print(data[4][i])
                        print(cur_rev_seq[i])
                        sys.exit(1)
                else:
                    rev_seq.append("")
                    rev_aa_score.append([-1])
                    rev_pep_score.append(-1)
                    rev_mz.append(0.0)
    outputf=open(opt.out,"w")
    outputf.write("True_seq\tseq\taa_score\tpep_score\trev_seq\trev_aa_score\trev_pep_score\ttrue_mz\tcharge\tforward_mz\trev_mz\n")
    for tt, tmp_seq, tmp_score, tmp_pep_score, tmp_rev_seq, tmp_rev_score, tmp_rev_p_score, tmp_true_mz, tmp_charge, tmp_forward_mz, tmp_rev_mz in zip(true_seq, seq, aa_score, pep_score, rev_seq, rev_aa_score, rev_pep_score,true_mz, charge, forward_mz, rev_mz):
        if tmp_pep_score==-1 and tmp_rev_p_score==-1:
            continue
        tmp=tt+"\t"+tmp_seq+"\t"
        for ss in tmp_score:
            tmp+=str(ss)+","
        tmp+="\t"+str(tmp_pep_score)
        tmp+="\t"+tmp_rev_seq+"\t"
        for ss in tmp_rev_score:
            tmp+=str(ss)+","
        tmp+="\t"+str(tmp_rev_p_score)
        tmp+="\t"+str(tmp_true_mz)
        tmp+="\t"+str(tmp_charge)
        tmp+="\t"+str(tmp_forward_mz)
        tmp+="\t"+str(tmp_rev_mz)+"\n"

        outputf.write(tmp)
    outputf.close()

    print(f"end evaluation: {time.ctime()}")


if __name__ == "__main__":
    opt = ownutils.parse_opt()
    if opt.type=="train":
        train(opt)
    elif opt.type=="gen":
        generation(opt)


    else:
        print("Error: wrong type")
        sys.exit(0)



