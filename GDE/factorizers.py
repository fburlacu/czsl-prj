'''Objects for decomposing embeddings of attribute-object compositions.'''

import torch
import numpy as np
import torch.nn.functional as F

from sphere import (calculate_intrinstic_mean,
                    logarithmic_map,
                    exponential_map,
                    parallel_transport)


def weighted_mean(embeddings, weights=None):
    '''Calculate the weighted mean of `embeddings`. The `weights` are normalized.'''
    if weights is None:
        return embeddings.mean(dim=0)
    else:
        return weights @ embeddings / weights.sum()


def compute_group_means(embeddings, group_ids, unique_groups, weights=None):
    '''
    Computes the mean vector for each group in unique_groups.
    Vector embeddings[i] and embeddings[j] belongs to the same group iff group_ids[i]=group_ids[j].
    '''
    group_id2idx= {id: [] for id in unique_groups}
    for i, group_id in enumerate(group_ids):
        group_id2idx[group_id].append(i)
    
    means = []
    for id in unique_groups:
        idx = group_id2idx[id]

        group_weights = None if weights is None else weights[idx]
        group_mean = weighted_mean(embeddings[idx], group_weights)
        means.append(group_mean)

    return torch.stack(means)

def compute_cond_means(embeddings, all_triplets_cond, attr, obj, weights=None):
    matching_indices = []
    for i, (attr1, attr2, triplet_obj) in enumerate(all_triplets_cond):
        if (attr2 == attr or attr1 == attr) and triplet_obj == obj:
            matching_indices.append(i)
    
    if not matching_indices:
        return None  # or raise an error if no matches found
    
    matching_indices = torch.tensor(matching_indices, device=embeddings.device)
    group_weights = None if weights is None else weights[matching_indices]
    cond_mean = weighted_mean(embeddings[matching_indices], group_weights)
    
    return cond_mean

def compute_cond_means_obj(embeddings, all_triplets_gt, obj, weights=None):
    matching_indices = []
    for i, (triplet_attr1, triplet_attr2, triplet_obj) in enumerate(all_triplets_gt):
        if triplet_obj == obj:
            matching_indices.append(i)
    
    if not matching_indices:
        return None
    
    matching_indices = torch.tensor(matching_indices, device=embeddings.device)
    group_weights = None if weights is None else weights[matching_indices]
    cond_mean = weighted_mean(embeddings[matching_indices], group_weights)
    
    return cond_mean

def compute_all_obj_means(embeddings, all_triplets_gt, attr1, attr2, weights=None):
    objs = [triplet[2] for triplet in all_triplets_gt]
    unique_objs = sorted(set(objs))
    
    obj_means = {}
    for obj in unique_objs:
        obj_mean = compute_cond_means_obj(embeddings, all_triplets_gt, attr1=None, attr2=None, obj=obj, weights=weights)
        if obj_mean is not None:
            obj_means[obj] = obj_mean
    
    return obj_means


# def compute_attr_obj_means(embeddings, all_pairs_gt, centered=True, weights=None):
#     '''
#     Computes mean for each attribute and object.
#     If two or more embeddings have the same pair, a the denoising step is performed first.
#     `weights` gives the weight distribution within pair. If None, uniform weights are used.
#     '''

#     mean_all = weighted_mean(embeddings, weights)
    
#     attrs, objs = zip(*all_pairs_gt)
#     attr_means = compute_group_means(embeddings, attrs, sorted(set(attrs)), weights)  # sorted wrt unique attrs
#     obj_means = compute_group_means(embeddings, objs, sorted(set(objs)), weights)     # sorted wrt unique objs
    
#     if centered:
#         attr_means = attr_means - mean_all
#         obj_means = obj_means - mean_all

#     return mean_all, attr_means, obj_means

def compute_attr1_attr2_obj_means(embeddings, all_triplets_gt, centered=True, weights=None):
    mean_all = weighted_mean(embeddings, weights)
    
    attrs1, attrs2, objs = zip(*all_triplets_gt)
    attr1_means = compute_group_means(embeddings, attrs1, sorted(set(attrs1)), weights)
    attr2_means = compute_group_means(embeddings, attrs2, sorted(set(attrs2)), weights)
    obj_means   = compute_group_means(embeddings, objs, sorted(set(objs)), weights)
    
    if centered:
        attr1_means = attr1_means - mean_all
        attr2_means = attr2_means - mean_all
        obj_means = obj_means - mean_all

    return mean_all, attr1_means, attr2_means, obj_means

# #list of all object means
# def compute_obj_means(embeddings, all_triplets_gt, weights = None):
#     object_list = [triplet[2] for triplet in all_triplets_gt]  #all the objects
#     unique_objects = sorted(set(object_list)) #list of unique objects

#     list_of_object_mean = []
#     for obj in unique_objects:
#         object_mean  = compute_cond_means_obj(embeddings, all_triplets_gt, obj, weights)
#         list_of_object_mean.append(object_mean)
#     return torch.stack(object_mean), unique_objects



    


# def compute_attr_means(embeddings, all_triplets_gt, obj_, weights):
#     required_attr1 = []
#     required_attr2 = []
#     required_idx   = []  
#     for i, (attr1, attr2, obj) in enumerate(all_triplets_gt):
#         if obj == obj_:
#             required_attr1.append(attr1)
#             required_attr2.append(attr2)
#             required_idx.append(i)
#     filtered_embeddings = embeddings[torch.tensor(required_idx)]

#     unq_attr1 = sorted(set(required_attr1))
#     attr1_obj_mean = compute_group_means(filtered_embeddings, required_attr1, unq_attr1, weights)


#     unq_attr2 = sorted(set(required_attr2))
#     attr2_obj_mean = compute_group_means(filtered_embeddings, required_attr2, unq_attr2, weights)

#     return attr1_obj_mean, attr2_obj_mean, unq_attr1, unq_attr2



# Factorizers

class CompositionalFactorizer:

    def __init__(self, embs_for_IW, all_triplets_gt, weights=None):
        '''
        Class that represents a compositional structure for a set of embeddings.
        Input:
            dataset: dataset of the embeddings
            embs_for_IW: embeddings used to compute the Ideal Words (primitive directions in the optimal decomposition)
            all_pair_gt: (attr, obj) label for `embs_for_IW`
            weights: weights assigned to the `embs_for_IW`, if `None` uniform weights are used. Weights are automatically normalized within pair.
        '''
        self.device = embs_for_IW.device
        # self.all_pairs_gt = all_pairs_gt
        self.all_triplets_gt = all_triplets_gt
        self.embs_for_IW = embs_for_IW
        self.weights = weights

        attrs1, attrs2, objs = zip(*all_triplets_gt)
        self.attrs1 = sorted(set(attrs1))
        self.attrs2 = sorted(set(attrs2))  #new new
        self.objs = sorted(set(objs))

        self.attr1_idx = {attr1: idx for idx, attr1 in enumerate(self.attrs1)}
        self.attr2_idx = {attr2: idx for idx, attr2 in enumerate(self.attrs2)}
        self.obj2idx   = {obj: idx for idx, obj in enumerate(self.objs)}

        # Compute IW for attrs and objs in dataset
        self.context, self.attr1_IW, self.attr2_IW, self.obj_IW = self.compute_ideal_words(
            embeddings=embs_for_IW,
            all_triplets_gt=all_triplets_gt,
            weights=weights
        )

    def compute_ideal_words(self, embeddings, all_pairs_gt):
        '''
        Extracts ideal words from `embeddings` labeled with `all_pairs_gt`.
        '''
        raise(NotImplementedError)
    
    def combine_ideal_words(self, *ideal_words, context=None):
        '''
        Combines ideal words using `context` as center.
        If context is `None`, `self.context` is used.
        '''
        raise(NotImplementedError)
    
    def get_attr1_IW(self, attr1):
        attr1_idx = self.attr1_idx[attr1]
        return self.attr1_IW[attr1_idx]
    
    def get_attr2_IW(self, attr2):
        attr2_idx = self.attr2_idx[attr2]
        return self.attr2_IW[attr2_idx]
    
    def get_obj_IW(self, obj):
        obj_idx = self.obj2idx[obj]
        return self.obj_IW[obj_idx]
    


    def compute_attr1_given_obj(self, attr1, obj):
        a1_iw  = self.get_attr1_IW(attr1)
        obj_iw = self.get_obj_IW(obj)

        n_a2       = len(self.attrs2)
        a2_IW_all  = self.attr2_IW                                    
        a1_iw_matrix  = a1_iw.unsqueeze(0).expand(n_a2, -1)            
        obj_iw_matrix = obj_iw.unsqueeze(0).expand(n_a2, -1)           

        composed = self.combine_ideal_words(
            a1_iw_matrix, a2_IW_all, obj_iw_matrix
        )                                                              

        return composed.mean(dim=0)  
    


    def compute_attr2_given_obj(self, attr2, obj):
 
        a2_iw  = self.get_attr2_IW(attr2)
        obj_iw = self.get_obj_IW(obj)

        n_a1       = len(self.attrs1)
        a1_IW_all  = self.attr1_IW                                    
        a2_iw_matrix  = a2_iw.unsqueeze(0).expand(n_a1, -1)             
        obj_iw_matrix = obj_iw.unsqueeze(0).expand(n_a1, -1)            

        composed = self.combine_ideal_words(
            a1_IW_all, a2_iw_matrix, obj_iw_matrix
        )                                                             

        return composed.mean(dim=0) 
    

    # def compute_ideal_words_approximation(self, target_pairs):
    #     target_attr_idx = torch.tensor(
    #         [self.attr2idx[attr] for attr, _ in target_pairs],
    #         device=self.device)
    #     target_obj_idx = torch.tensor(
    #         [self.obj2idx[obj] for _, obj in target_pairs],
    #         device=self.device)
        
    #     # Select attr_IW and obj_IW for target pairs
    #     attrIW_target = self.attr_IW[target_attr_idx]
    #     objIW_target = self.obj_IW[target_obj_idx]

    #     # Compute IW approximation for target pairs
    #     target_IWapprox = self.combine_ideal_words(attrIW_target, objIW_target)

    #     return target_IWapprox

    def compute_obj_means(self, embeddings, all_triplets_gt, weights=None):
        object_list = [triplet[2] for triplet in all_triplets_gt]
        unique_objects = sorted(set(object_list))

        list_of_object_mean = []
        for obj in unique_objects:
            object_mean  = compute_cond_means_obj(embeddings, all_triplets_gt, obj, weights)
            list_of_object_mean.append(object_mean)
        return torch.stack(list_of_object_mean), unique_objects

    def compute_attr_means(self, obj_):
        required_attr1 = []
        required_attr2 = []
        required_idx   = []  
        for i, (attr1, attr2, obj) in enumerate(self.all_triplets_gt):
            if obj == obj_:
                required_attr1.append(attr1)
                required_attr2.append(attr2)
                required_idx.append(i)
        
        # Filter embeddings and weights to match the specific object
        filtered_embeddings = self.embs_for_IW[torch.tensor(required_idx)]
        filtered_weights = self.weights[torch.tensor(required_idx)] if self.weights is not None else None

        unq_attr1 = sorted(set(required_attr1))
        attr1_obj_mean = compute_group_means(filtered_embeddings, required_attr1, unq_attr1, filtered_weights)

        unq_attr2 = sorted(set(required_attr2))
        attr2_obj_mean = compute_group_means(filtered_embeddings, required_attr2, unq_attr2, filtered_weights)

        return attr1_obj_mean, attr2_obj_mean, unq_attr1, unq_attr2

    def compute_ideal_words_approximation(self, target_triplet):  #handles triplets
        target_attr1_idx = torch.tensor(
            [self.attr1_idx[attr1] for attr1, _ in target_triplet],
            device=self.device)
        
        target_attr2_idx = torch.tensor(
            [self.attr2_idx[attr2] for attr2, _ in target_triplet],
            device=self.device)
        
        target_obj_idx = torch.tensor(
            [self.obj2idx[obj] for _, obj in target_triplet],
            device=self.device)
        
        # Select attr_IW and obj_IW for target pairs
        attr1IW_target = self.attr1_IW[target_attr1_idx]
        attr2IW_target = self.attr2_IW[target_attr2_idx]
        objIW_target = self.obj_IW[target_obj_idx]

        # Compute IW approximation for target pairs
        target_IWapprox = self.combine_ideal_words(attr1IW_target, attr2IW_target, objIW_target)

        return target_IWapprox
        
    def __str__(self) -> str:
        return self.name

    

class LDE(CompositionalFactorizer):
    name = 'LDE'

    def compute_ideal_words(self, embeddings, all_triplets_gt, weights):
        return compute_attr1_attr2_obj_means(embeddings, all_triplets_gt, weights=weights)

    def combine_ideal_words(self, *ideal_words, context=None):
        if context is None:
            context = self.context
        ideal_words = torch.stack(ideal_words)
        return context + torch.sum(ideal_words, dim=0)
    
    # def get_denoised_pair(self):
    #     unique_pairs = list(set(self.all_pairs_gt))
    #     denoised_pair = compute_group_means(self.embs_for_IW, self.all_pairs_gt, unique_pairs)
    #     return unique_pairs, denoised_pair

    def get_denoised_triplets(self):  #handles triplets
        unique_triplets = list(set(self.all_triplets_gt))
        denoised_triplets = compute_group_means(self.embs_for_IW, self.all_triplets_gt, unique_triplets)
        return unique_triplets, denoised_triplets


class GDE(CompositionalFactorizer):  #important
    name = 'GDE'
    
    def compute_ideal_words(self, embeddings, all_triplets_gt, weights):
        intrinsic_mean = calculate_intrinstic_mean(embeddings, weights, init='normalized mean')  # mu

        # 1) Map embedding to the tangent space T_muS^n
        embeddings_T = logarithmic_map(intrinsic_mean, embeddings)

        # 2) Compute IW on the tangent space
        v_c, attr1_IW, attr2_IW, obj_IW = compute_attr1_attr2_obj_means(embeddings_T, all_triplets_gt, weights=weights)
        assert torch.norm(v_c, p=2) < 1e-5 # should be v_c=0  (WILL LOOK INTO THIS)

        context = intrinsic_mean
        return context, attr1_IW, attr2_IW , obj_IW

    def combine_ideal_words(self, *ideal_words, context=None):
        if context is None:
            original_contex = True
            context = self.context
        else:
            original_contex = False

        # 3) Combine ideal words in the tangent plane
        ideal_words = torch.stack(ideal_words)
        embs_approx_T = torch.sum(ideal_words, dim=0)

        # 4) Map the obtained approximation back to the sphere
        if original_contex:
            embs_approx = exponential_map(context, embs_approx_T)
        else:
            # If context is not mu, we need to transport embs_approx_T from T_muS^n to T_contextS^n
            embs_approx_T_transported = parallel_transport(self.context, context, embs_approx_T)
            embs_approx = exponential_map(context, embs_approx_T_transported)
        
        return embs_approx

    # def get_denoised_pair(self):
    #     unique_pairs = list(set(self.all_pairs_gt))
    #     embs_T = logarithmic_map(self.context, self.embs_for_IW)
    #     denoised_pair_T = compute_group_means(embs_T, self.all_pairs_gt, unique_pairs)
    #     denoised_pair = exponential_map(self.context, denoised_pair_T)
    #     return unique_pairs, denoised_pair

    def get_denoised_pair(self):
        unique_triplets = list(set(self.all_triplets_gt))
        embs_T = logarithmic_map(self.context, self.embs_for_IW)
        denoised_triplets_T = compute_group_means(embs_T, self.all_triplets_gt, unique_triplets)
        denoised_triplets = exponential_map(self.context, denoised_triplets_T)
        return unique_triplets, denoised_triplets



FACTORIZERS = {
    'LDE': LDE,
    'GDE': GDE,
}