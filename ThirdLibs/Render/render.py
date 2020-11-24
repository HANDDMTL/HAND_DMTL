
import cv2
import sys
from os import path as osp
import math
import numpy as np
import glob
import torch
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo

from scipy.spatial import Delaunay
from PIL import Image
from scipy.spatial.transform import Rotation

cur_dir = osp.dirname(osp.abspath(__file__))

def create_program(s_verts, s_frags):
    #compile vertex shader
    s_id = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(s_id, s_verts)
    glCompileShader(s_id)
    rt_code = glGetShaderiv(s_id, GL_COMPILE_STATUS)
    if rt_code==0:
        print('invalid {}, msg {}'.format('VERTEX_SHADER', glGetShaderInfoLog(s_id)))
        sys.exit(0)
    
    #compile fragment shader
    f_id = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(f_id, s_frags)
    glCompileShader(f_id)
    rt_code = glGetShaderiv(f_id, GL_COMPILE_STATUS)
    if rt_code == 0:
        print('invalid {}, msg {}'.format('FRAGMENT_SHADER', glGetShaderInfoLog(f_id)))
        sys.exit(0)

    #create program
    proc_id = glCreateProgram()
    glAttachShader(proc_id, s_id)
    glAttachShader(proc_id, f_id)
    glLinkProgram(proc_id)
    rt_code = glGetProgramiv(proc_id, GL_LINK_STATUS)
    print("link program success [%s]" %(rt_code))
    return proc_id

def computeNormals(verts,faces):
    v0=verts[faces[:,0]]
    v1=verts[faces[:,1]]
    v2=verts[faces[:,2]]
    dv1=v1-v0
    dv2=v2-v0
    dn=np.cross(dv2,dv1)
    normals=np.zeros(verts.shape,np.float32)
    normals[faces[:,0]]=normals[faces[:,0]]+dn
    normals[faces[:,1]]=normals[faces[:,1]]+dn
    normals[faces[:,2]]=normals[faces[:,2]]+dn
    normals=normals/np.repeat(np.linalg.norm(normals,axis=1).reshape(-1,1), 3, axis=1)
    return normals
    
def get_pers_proj_matrix(fov, asp, zn, zf, s, w, h, dw, dh, img_size=256):
    CotValue = 1.0/math.tan(fov*3.1415926/180.0)
    proj_mat = np.zeros([4,4]).astype(np.float)
    dif = zn-zf
    a = -zn/dif
    b = zn*zf/dif
    proj_mat[0][0] = s*img_size/w * CotValue
    proj_mat[1][1] = s*img_size/h * CotValue
    proj_mat[2][2] = a
    proj_mat[2][3] = b
    proj_mat[3][2] = 1.0
    proj_mat[0][2] = (img_size*s+2*dw-w)/w
    proj_mat[1][2] = (img_size*s+2*dh-h)/h

    return proj_mat

def get_orth_proj_matrix(fov, asp, zn, zf, s, w, h, t_xyz, dw, dh, img_size=256):
    proj_mat = np.zeros([4,4]).astype(np.float)
    dif = zf-zn
    a = 1/dif
    b = -zn/dif
    proj_mat[0][0] = t_xyz[0]*s*img_size/w
    proj_mat[1][1] = t_xyz[0]*s*img_size/h
    proj_mat[2][2] = a
    proj_mat[2][3] = b
    proj_mat[3][3] = 1.0

    proj_mat[0][3] = (t_xyz[0]*s*img_size*t_xyz[1]+s*img_size+2*dw-w)/w
    proj_mat[1][3] = (t_xyz[0]*s*img_size*t_xyz[2]+s*img_size+2*dh-h)/h

    return proj_mat

class RenderFace(object):
    def __init__(self, w=1024, h=1024, img_size=256):
        self.w, self.h = w, h
        self.create_window()
        self.s_verts = '''
            attribute vec3 position;
            attribute vec3 normal;
            varying vec3 outNormal;
            varying vec3 outpos;
            varying vec3 color;
            uniform mat4 mvp;
            void main() {
                outNormal = -normal;
                outpos = position;
                gl_Position = mvp * vec4(position, 1.0);
            }
        '''

        self.f_verts = '''
            varying vec3 outNormal;
            varying vec3 outpos;
            uniform int mode;
            void main () {
                vec3 norm = normalize(outNormal);
                vec3 l = vec3(0.2, 0.2, 1.0);
                l = normalize(l);
                vec3 r = l - 2.0 * norm * dot(norm, l);
                float dif = max(dot(-l, norm), 0.0)/1.3;
                vec3 v = -normalize(-outpos);
                float spec = pow(max(dot(v, r), 0.0), 5); 
                float amb = 0.1;
                float c = dif + spec + amb;
                if (mode < 2) {
                    gl_FragColor.rgba = vec4(c, c, c, 1.0);
                } else {
                    gl_FragColor.rgba = vec4(1.0, 0.0, 0.0, 1.0);
                }
            }
        '''

        self.proc_id = create_program(self.s_verts, self.f_verts)
        self.faces = None
        self.img_size = img_size

        self.tracking_mouse = False
        self.pre_x = 0
        self.pre_y = 0
        self.cur_x = 0
        self.cur_y = 0

    def setup(self, verts, joints, faces, t_xyz, dw, dh, s, fov):
        w, h = self.w, self.h
        self.faces = faces
        self.gl_faces     = vbo.VBO(self.faces.reshape(-1).astype(np.uint32), target = GL_ELEMENT_ARRAY_BUFFER)
        self.nface     = self.faces.shape[0] * 3
        self.proj_mat = get_pers_proj_matrix(fov, asp=self.w/self.h, zn=0.001, zf=10, s=s, w=self.w, h=self.h, dw=dw, dh=dh, img_size=self.img_size)
        
        self.verts, self.joints, self.t_xyz = verts, joints, t_xyz

    def start_render(self):
        pass

    def create_window(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH | GLUT_STENCIL)
        glutInitWindowSize (self.w, self.h)
        glutCreateWindow('TexGen')
        glutReshapeWindow(self.w, self.h)
        glutReshapeFunc(self._cb_reshape_)
        glutDisplayFunc(self._cb_display)
        glutKeyboardFunc(self._cb_keyboard)
        glutMotionFunc(self._cb_mouse_motion)
        glutMouseFunc(self._cb_mouse_clk)

    def _cb_reshape_(self, width, height):
        glViewport(0, 0, width, height)

    def _cb_mouse_clk(self, button, state, x, y):
        if button == GLUT_LEFT_BUTTON and button == GLUT_DOWN:
            self.tracking_mouse = True
            self.cur_x, self.cur_y = x, y
            self.pre_x, self.pre_y = x, y
        else:
            self.tracking_mouse = False
            self.cur_x, self.cur_y = 0, 0
            self.pre_x, self.pre_y = 0, 0

    def _cb_mouse_motion(self, x, y):
        if self.tracking_mouse:
            self.cur_x, self.cur_y = x, y
        
        self._cb_display()

    def _cb_draw(self):
        glUseProgram(self.proc_id)
        glEnable(GL_CULL_FACE)
        glFrontFace(GL_CW)
        glDisable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        mode_loc = glGetUniformLocation(self.proc_id, 'mode')
        glUniform1i(mode_loc, 0)
        self.pos.bind()
        pos_loc = glGetAttribLocation(self.proc_id, 'position')
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc, 3, GL_FLOAT,False, 0, None)
        
        self.normals.bind()
        norm_loc = glGetAttribLocation(self.proc_id, 'normal')
        glEnableVertexAttribArray(norm_loc)
        glVertexAttribPointer(norm_loc, 3, GL_FLOAT,False, 0, None)

        mat_loc = glGetUniformLocation(self.proc_id, 'mvp')
        glUniformMatrix4fv(mat_loc, 1, True, self.proj_mat)

        self.gl_faces.bind()
        glDrawElements(GL_TRIANGLES, self.nface, GL_UNSIGNED_INT, None)

        glDisable(GL_DEPTH_TEST)
        self.gl_joints_position.bind()
        pos_loc = glGetAttribLocation(self.proc_id, 'position')
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc, 3, GL_FLOAT,False, 0, None)
        glPointSize(2.5)
        mode_loc = glGetUniformLocation(self.proc_id, 'mode')
        glUniform1i(mode_loc, 10)
        self.gl_joints_indexs.bind()
        glDrawElements(GL_POINTS, self.joints.shape[0], GL_UNSIGNED_INT, None)


    def _cb_display(self):
        dy = self.cur_y - self.pre_y
        dx = self.cur_x - self.pre_x
        rot_mat = Rotation.from_euler('zyx', [[0, dx/5, dy/5]], degrees=True).as_dcm()[0]

        verts = np.matmul(self.verts, rot_mat)
        joints = np.matmul(self.joints, rot_mat)

        norms = computeNormals(verts, self.faces)
        
        self.normals   = vbo.VBO(norms.astype(np.float32).reshape(-1, 3))
        self.pos       = vbo.VBO(verts.astype(np.float32).reshape(-1, 3))

        self.gl_joints_position =  vbo.VBO(joints.astype(np.float32).reshape(-1, 3))
        self.gl_joints_indexs   =  vbo.VBO(np.arange(joints.shape[0]).astype(np.uint32), target = GL_ELEMENT_ARRAY_BUFFER)

        self._cb_draw()
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, self.w, self.h, GL_BGRA, GL_UNSIGNED_BYTE)
        image = np.array(Image.frombytes("RGBA", (self.w, self.h), data))
        glutSwapBuffers()
        return image

    def _cb_keyboard(self, keycode, x, y):
        print(keycode)
        if (keycode == b'q'):
            sys.exit()

    def _message_loop(self):
        glutMainLoop()