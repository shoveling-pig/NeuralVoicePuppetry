import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

import soft_renderer as sr

N_EXPRESSIONS=76

class MorphableModel(nn.Module):
    def __init__(self, filename_average=''):
        super(MorphableModel, self).__init__()

        print('Load Morphable Model (Basel)')

        #filename_mesh = os.path.join(opt.dataroot, opt.phase + '/average_model.obj')
        filename_mesh = filename_average
        if filename_average=='':
            print('use default identity')
            filename_mesh = './BaselModel/average.obj'
        mesh = sr.Mesh.from_obj(filename_mesh, normalization=False, load_texture=False)
        self.average_vertices = mesh.vertices[0]
        self.faces = mesh.faces[0]
        self.average_vertices = self.average_vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
        self.faces = self.faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]
        self.textures = mesh.textures

        self.num_vertices = self.average_vertices.shape[1]
        self.num_faces = self.faces.shape[1]
        print('vertices:', self.average_vertices.shape)
        print('faces:', self.faces.shape)

        ## basis function
        self.expression_basis = np.load('/data3/shovelingpig/STV/NeuralVoicePuppetry/Audio2ExpressionNet/Training Code/BaselModel/ExpressionBasis.npy')
        self.expression_basis = np.reshape(self.expression_basis,  (self.num_vertices, 3, N_EXPRESSIONS))
        self.expression_basis = torch.tensor(self.expression_basis.astype(np.float32)).cuda() 
        self.expression_basis = torch.transpose(self.expression_basis, 0, 1) # 3 x num_vertices x N_EXPRESSIONS


        #texture_size = 2
        #self.textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
        #print('textures:', self.textures.shape)

        ## ## debug
        ## zeroExpr = torch.zeros(1, N_EXPRESSIONS, dtype=torch.float32).cuda()
        ## self.morph(zeroExpr)
        ## self.save_model_to_obj_file('model_zero_expression.obj')
        ## ##
        ## onesExpr = torch.ones(1, N_EXPRESSIONS, dtype=torch.float32).cuda()
        ## self.morph(onesExpr)
        ## self.save_model_to_obj_file('model_ones_expression.obj')
        ## exit()
        ## ##

        # default expression
        zeroExpr = torch.zeros(1, N_EXPRESSIONS, dtype=torch.float32).cuda()
        self.morph(zeroExpr)


    def save_model_to_obj_file(self, filename, mask=None):
        faces_cpu = self.faces.detach().cpu().numpy()
        vertices_cpu = self.vertices.detach().cpu().numpy()

        mask_cpu = None
        if not type(mask) == type(None):
            mask_cpu = mask.detach().cpu().numpy()

        f = open(filename, 'w')
        if type(mask) == type(None):
            for i in range(0, self.num_vertices):
                f.write('v '  + str(vertices_cpu[0, i, 0]) + ' ' + str(vertices_cpu[0, i, 1]) + ' ' + str(vertices_cpu[0, i, 2]) + '\n')
        else:
            for i in range(0, self.num_vertices):
                f.write('v '  + str(vertices_cpu[0, i, 0]) + ' ' + str(vertices_cpu[0, i, 1]) + ' ' + str(vertices_cpu[0, i, 2]) + ' ' +  str(mask_cpu[i]) + ' ' +  str(mask_cpu[i]) + ' ' +  str(mask_cpu[i])  + ' 1'+ '\n')

        for i in range(0, self.num_faces):
            f.write('f '  + str(faces_cpu[0, i, 0]+1) + '// ' + str(faces_cpu[0, i, 1]+1) + '// ' + str(faces_cpu[0, i, 2]+1) + '//\n')

        f.close()

    def compute_expression_delta(self, expressions):
        return torch.transpose(torch.matmul(self.expression_basis, torch.transpose(expressions, 0, 1)), 0, 2) # note that matmul wants to have this order:  (a x b x c) x (c x m) => (a x b x m)

    def morph(self, expressions):
        self.vertices = self.average_vertices + self.compute_expression_delta(expressions)
        return self.vertices



    def LoadMask(self, filename=''):
        if filename=='':
            print('use default mask')
            filename = './BaselModel/mask/defaultMask_mouth.obj'

        mask = np.zeros(self.num_vertices)
        file = open(filename, 'r')
        i=0
        for line in file:
            if line[0] == 'v':
                floats = [float(x) for x in line[1:].split()]
                if floats[3] == 1.0 and floats[4] == 0.0 and floats[5] == 0.0:
                    mask[i] = 1.0
                i += 1      
        file.close()
        return  torch.tensor(mask.astype(np.float32)).cuda() 