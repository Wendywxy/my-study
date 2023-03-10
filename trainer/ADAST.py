import torch
import torch.nn as nn
from trainer.training_evaluation import model_evaluate, val_self_training
from utils import calc_similiar_penalty
from models.models import Discriminator, AttnSleep,Classifier #Self_Attn, cnn_feature_extractor, Classifier, Self_Attn
from dataloader.dataloader import *
#from parse_config import ConfigParser
import math

def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

#Added by wendy start-----
def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5]

    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * factor, 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight

def weighted_CrossEntropyLoss(output, target, classes_weights, device):
    cr = nn.CrossEntropyLoss(weight=torch.tensor(classes_weights).to(device))
    return cr(output, target)
#Added by wendy end-----
#def cross_domain_train(src_train_dl, trg_train_dl, trg_valid_dl,
#                       src_id, trg_id,
#                       device, logger, configs, args, param_config):
def cross_domain_train(src_train_dl, trg_train_dl, trg_valid_dl,
                       src_id, trg_id,
                       device, logger, configs, args, param_config): #Use attsleep loss
    # source model network.
    model_configs = configs.base_model

    # build feature extractor architecture, initialize weights, then print to console
    model1 = AttnSleep()
    model1.apply(weights_init_normal)
    logger.info(model1)

    # Split the first conv block in model
#    feature_extractor = cnn_feature_extractor(model_configs).float().to(device)
    feature_extractor1 = AttnSleep().float().to(device)
    classifier_1 = Classifier(model_configs).float().to(device)
    classifier_2 = Classifier(model_configs).float().to(device)

    feature_discriminator = Discriminator(model_configs).to(device)
    '''
    src_att = Self_Attn(model_configs.final_out_channels).to(device)
    trg_att = Self_Attn(model_configs.final_out_channels).to(device)
    '''
    # loss functions
    disc_criterion = nn.BCEWithLogitsLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer.
    optimizer_encoder = torch.optim.Adam(
        list(feature_extractor1.parameters()) + list(classifier_1.parameters()) + list(classifier_2.parameters()),
            lr=configs.lr, betas=(configs.beta1, configs.beta2),
        weight_decay=configs.weight_decay)

    optimizer_disc = torch.optim.Adam(feature_discriminator.parameters() , lr=configs.lr,
                                      betas=(configs.beta1, configs.beta2), weight_decay=configs.weight_decay)
    #acc = 0
    for round_idx in range(param_config.self_training_iterations):

        scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=configs.step_size,
                                                            gamma=configs.gamma)
        if round_idx == 0:
            trg_clf_wt = 0
            src_clf_wt = param_config.src_clf_wt
        else:
            src_clf_wt = param_config.src_clf_wt * 0.1
            trg_clf_wt = param_config.trg_clf_wt
        logger.debug("trg_id"+ trg_id)
        logger.debug("round idx"+str(round_idx))
        logger.debug("feature extractor"+str(feature_extractor1))
        logger.debug("classifier1"+str(classifier_1))
        logger.debug("classifier2"+str(classifier_2))  
        # generate pseudo labels
        val_self_training((feature_extractor1,(classifier_1,classifier_2)),trg_train_dl,device,src_id,trg_id,round_idx,args)

        file_name = f"pseudo_train_{src_id}_to_{trg_id}_round_{round_idx}.pt"
        pseudo_trg_train_dataset = torch.load(os.path.join(args.home_path, "data", file_name))

        # Loading datasets
        pseudo_trg_train_dataset = Load_Dataset(pseudo_trg_train_dataset)

        # Dataloader for target pseudo labels
        pseudo_trg_train_dl = torch.utils.data.DataLoader(dataset=pseudo_trg_train_dataset,
                                                          batch_size=configs.batch_size,
                                                          shuffle=True, drop_last=True,
                                                          num_workers=0)

        # training..
        for epoch in range(1, configs.num_epoch + 1):
            joint_loaders = enumerate(zip(src_train_dl, pseudo_trg_train_dl))
            feature_extractor1.train()
            classifier_1.train()
            classifier_2.train()

            '''
            src_att.train()
            trg_att.train()
            '''
            feature_discriminator.train()

            for step, ((src_data, src_labels), (trg_data, pseudo_trg_labels)) in joint_loaders:
                src_data, src_labels, trg_data, pseudo_trg_labels = src_data.float().to(device), src_labels.long().to(
                    device), trg_data.float().to(device), pseudo_trg_labels.long().to(device)

                for param in feature_discriminator.parameters():
                    param.requires_grad = True

                # pass data through the source model network.
                src_feat = feature_extractor1(src_data)
                #src_feat = src_att(src_feat)
                src_pred = classifier_1(src_feat)
                src_pred_2 = classifier_2(src_feat)

                # pass data through the target model network.
                #trg_feat = trg_att(trg_feat)
                trg_feat = feature_extractor1(trg_data)
                trg_pred = classifier_1(trg_feat)
                trg_pred_2 = classifier_2(trg_feat)

                # concatenate source and target features
                concat_feat = torch.cat((src_feat, trg_feat), dim=0)

                # predict the domain label by the discirminator network
                concat_pred = feature_discriminator(concat_feat.detach())

                # prepare real labels for the training the discriminator
                disc_src_labels = torch.ones(src_feat.size(0)).long().to(device)
                disc_trg_label = torch.zeros(trg_feat.size(0)).long().to(device)
                label_concat = torch.cat((disc_src_labels, disc_trg_label), 0)

                # Discriminator Loss
                loss_disc = disc_criterion(concat_pred.squeeze(), label_concat.float())

                optimizer_disc.zero_grad()
                loss_disc.backward()
                optimizer_disc.step()

                for param in feature_discriminator.parameters():
                    param.requires_grad = False

                trg_pred2 = feature_discriminator(trg_feat)
                src_pred2 = feature_discriminator(src_feat)

                # prepare fake labels
                fake_src_label = (torch.zeros(src_feat.size(0)).long()).to(device)
                fake_trg_label = (torch.ones(trg_feat.size(0)).long()).to(device)

                # Compute Adversarial Loss
                loss_adv = disc_criterion(torch.cat((trg_pred2.squeeze(), src_pred2.squeeze()), 0),
                                          torch.cat((fake_trg_label.float(), fake_src_label.float()), 0))

                # Compute Source Classification Loss
                src_clf_loss = criterion(src_pred, src_labels) + criterion(src_pred_2, src_labels)
                #src_class_weight = calc_class_weight(src_data_count)
                #src_clf_loss = weighted_CrossEntropyLoss(src_pred,src_labels,src_class_weight,device) + weighted_CrossEntropyLoss(src_pred_2,src_labels,src_class_weight,device)

                # Compute similarity penalty
                similarity_penalty = calc_similiar_penalty(classifier_1, classifier_2)

                # Compute target classification loss
                trg_clf_loss = criterion(trg_pred, pseudo_trg_labels) + criterion(trg_pred_2, pseudo_trg_labels)
                #trg_class_weight = calc_class_weight(trg_data_count)
                #trg_clf_loss = weighted_CrossEntropyLoss(trg_pred,pseudo_trg_labels,trg_class_weight,device) + weighted_CrossEntropyLoss(trg_pred_2,pseudo_trg_labels,trg_class_weight,device)

                # total loss calculations
                total_loss = param_config.disc_wt * loss_adv + \
                             src_clf_wt * src_clf_loss + \
                             param_config.similarity_wt * similarity_penalty + \
                             trg_clf_wt * trg_clf_loss

                optimizer_encoder.zero_grad()
                total_loss.backward()
                optimizer_encoder.step()

            if round_idx == 0:
                scheduler_encoder.step()

            # to print learning rate every epoch
            # for param_group in optimizer_encoder.param_groups:
            #     print(param_group['lr'])

            if epoch % 1 == 0:
			
			    #target_loss, target_score, _, _ = model_evaluate(
                #    (feature_extractor, trg_att, (classifier_1, classifier_2)), trg_valid_dl, device)
                #target_loss, target_score, _, _ = model_evaluate(
                #    (feature_extractor1,(classifier_1,classifier_2)),trg_valid_dl,device)
                target_loss, target_score, _, _ = model_evaluate(
                    (feature_extractor1,(classifier_1,classifier_2)),trg_valid_dl,device)

                logger.debug(f'[Epoch : {epoch}/{configs.num_epoch}]')
                logger.debug(
                    f'{args.da_method} Loss  : {target_loss:.4f}\t | \t{args.da_method} Accuracy  : {target_score:2.4f}')
                logger.debug(f'-------------------------------------')
                '''
                #To return the best model
                if acc < target_score:
                    acc = target_score
                    print("best acc",acc)
                    feat_best = feature_extractor1
                    class1_best = classifier_1
                    class2_best = classifier_2
                '''
    return (feature_extractor1,(classifier_1,classifier_2))
    #return (feat_best,(class1_best,class2_best))
