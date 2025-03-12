
import sys
from typing import Dict
from SmplObject import SmplObjects
import os
from scipy.spatial.transform import Rotation as R
import numpy as np

# try:
#     from FbxCommon import *
#     from fbx import *
# except ImportError:
#     print("Error: module FbxCommon failed to import.\n")
#     print("Copy the files located in the compatible sub-folder lib/python<version> into your python interpreter site-packages folder.")
#     import platform
#     if platform.system() == 'Windows' or platform.system() == 'Microsoft':
#         print('For example: copy ..\\..\\lib\\Python27_x64\\* C:\\Python27\\Lib\\site-packages')
#     elif platform.system() == 'Linux':
#         print('For example: cp ../../lib/Python27_x64/* /usr/local/lib/python2.7/site-packages')
#     elif platform.system() == 'Darwin':
#         print('For example: cp ../../lib/Python27_x64/* /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')



from FbxCommon import *
from fbx import *



# def DisplayGlobalTimeSettings(pGlobalSettings):
#     lTimeModes = [ "Default Mode", "Cinema", "PAL", "Frames 30", 
#         "NTSC Drop Frame", "Frames 50", "Frames 60",
#         "Frames 100", "Frames 120", "NTSC Full Frame", 
#         "Frames 30 Drop", "Frames 1000" ] 

#     DisplayString("Time Mode: ", lTimeModes[pGlobalSettings.GetTimeMode().value])

#     lTs = pGlobalSettings.GetTimelineDefaultTimeSpan()
#     lStart = lTs.GetStart()
#     lEnd   = lTs.GetStop()
#     DisplayString("Timeline default timespan: ")
#     lTmpStr=""
#     DisplayString("     Start: ", lStart.GetTimeString(lTmpStr, 10))
#     DisplayString("     Stop : ", lEnd.GetTimeString(lTmpStr, 10))

#     DisplayString("")
class FbxReadWrite(object):
    def __init__(self, fbx_source_path):
        # Prepare the FBX SDK.
        lSdkManager, lScene = InitializeSdkObjects()
        self.lSdkManager = lSdkManager
        self.lScene = lScene

        # Load the scene.
        # The example can take a FBX file as an argument.
        print("\nLoading File: {}".format(fbx_source_path))
        lResult = LoadScene(self.lSdkManager, self.lScene, fbx_source_path)

        

        if not lResult:
            raise Exception("An error occured while loading the scene :(")

    def _write_curve(self, lCurve:FbxAnimCurve, data:np.ndarray):
        """
        data: np.ndarray of (N, )
        """
        lKeyIndex = 0
        lTime = FbxTime()
        
        ## bug: FbxTime.eFrames60 is not defined --> use FbxTime.EMode.eFrames60
        lTime.SetGlobalTimeMode(FbxTime.EMode.eFrames60) # Set to fps=60

        data = np.squeeze(data)

        lCurve.KeyModifyBegin()
        for i in range(data.shape[0]):
            # lTime.SetFrame(i, FbxTime.eFrames60)
            lTime.SetSecondDouble(i / 60.0)  # NOTE: THIS LINE IS IMPORTANT. I don't know why it works but this line is necessary. Otherwise the output FBX does not have motion information
            
            lKeyIndex = lCurve.KeyAdd(lTime)[0]
            lCurve.KeySetValue(lKeyIndex, data[i])
            lCurve.KeySetInterpolation(lKeyIndex, FbxAnimCurveDef.EInterpolationType.eInterpolationCubic)
        lCurve.KeyModifyEnd()
        """
        FbxTime:
            __module__: fbx, __lt__: <slot wrapper '__lt__' of 'FbxTime' objects>, __le__: <slot wrapper '__le__' of 'FbxTime' objects>, __eq__: <slot wrapper '__eq__' of 'FbxTime' objects>, 
            __ne__: <slot wrapper '__ne__' of 'FbxTime' objects>, __gt__: <slot wrapper '__gt__' of 'FbxTime' objects>, __ge__: <slot wrapper '__ge__' of 'FbxTime' objects>, __add__: <slot wrapper '__add__' of 'FbxTime' objects>,
            __radd__: <slot wrapper '__radd__' of 'FbxTime' objects>, __sub__: <slot wrapper '__sub__' of 'FbxTime' objects>, __rsub__: <slot wrapper '__rsub__' of 'FbxTime' objects>, __mul__: <slot wrapper '__mul__' of 'FbxTime' objects>,
            __rmul__: <slot wrapper '__rmul__' of 'FbxTime' objects>, __iadd__: <slot wrapper '__iadd__' of 'FbxTime' objects>, __isub__: <slot wrapper '__isub__' of 'FbxTime' objects>, 
            __truediv__: <slot wrapper '__truediv__' of 'FbxTime' objects>, __rtruediv__: <slot wrapper '__rtruediv__' of 'FbxTime' objects>, __weakref__: <attribute '__weakref__' of 'FbxTime' objects>, __doc__: FbxTime(int = 0),
              __hash__: None, 
                ConvertFrameRateToTimeMode: <built-in method ConvertFrameRateToTimeMode>, 
                Get: <built-in method Get>, GetFieldCount: <built-in method GetFieldCount>, 
                GetFrameCount: <built-in method GetFrameCount>, 
                GetFrameCountPrecise: <built-in method GetFrameCountPrecise>, 
                GetFrameRate: <built-in method GetFrameRate>, GetFrameSeparator: 
                <built-in method GetFrameSeparator>, 
                GetFramedTime: <built-in method GetFramedTime>, 
                GetGlobalTimeMode: <built-in method GetGlobalTimeMode>, 
                GetGlobalTimeProtocol: <built-in method GetGlobalTimeProtocol>, 
                GetHourCount: <built-in method GetHourCount>, 
                GetMilliSeconds: <built-in method GetMilliSeconds>,
                GetMinuteCount: <built-in method GetMinuteCount>,
                GetResidual: <built-in method GetResidual>, 
                GetSecondCount: <built-in method GetSecondCount>,
                GetSecondDouble: <built-in method GetSecondDouble>, 
                GetTime: <built-in method GetTime>, 
                GetTimeString: <built-in method GetTimeString>, 
                IsDropFrame: <built-in method IsDropFrame>, 
                Set: <built-in method Set>, SetFrame: <built-in method SetFrame>, 
                SetFramePrecise: <built-in method SetFramePrecise>, 
                SetGlobalTimeMode: <built-in method SetGlobalTimeMode>, 
                SetGlobalTimeProtocol: <built-in method SetGlobalTimeProtocol>,
                SetMilliSeconds: <built-in method SetMilliSeconds>, 
                SetSecondDouble: <built-in method SetSecondDouble>, 
                SetTime: <built-in method SetTime>, 
                SetTimeString: <built-in method SetTimeString>, 
                EElement: <enum 'EElement'>, 
                EMode: <enum 'EMode'>, EProtocol: <enum 'EProtocol'>
        """
    
    def addAnimation(self, pkl_filename:str, smpl_params:Dict, verbose:bool = False):
        lScene = self.lScene

        # 0. Set fps to 60
        lGlobalSettings = lScene.GetGlobalSettings()
        print(dir(lGlobalSettings))
        """
        lGlobalSettings:
            ['ClassId', 'Clone', 'ConnectDstObject', 'ConnectDstProperty', 'ConnectSrcObject', 'ConnectSrcProperty', 'ContentDecrementLockCount',
            'ContentIncrementLockCount', 'ContentIsLoaded', 'ContentIsLocked', 'ContentLoad', 'ContentUnload', 'Create', 'Destroy', 
            'DisconnectAllDstObject', 'DisconnectAllSrcObject', 'DisconnectDstObject', 'DisconnectDstProperty', 'DisconnectSrcObject', 'DisconnectSrcProperty', 
            'ECloneType', 'EObjectFlag', 'FindDstObject', 'FindDstProperty', 'FindProperty', 'FindPropertyHierarchical', 'FindSrcObject', 'FindSrcProperty',
            'GetAllObjectFlags', 'GetAmbientColor', 'GetAxisSystem', 'GetClassId', 'GetClassRootProperty', 'GetDefaultCamera', 'GetDocument', 'GetDstObject', 
            'GetDstObjectCount', 'GetDstProperty', 'GetDstPropertyCount', 'GetFirstProperty', 'GetInitialName', 'GetName', 'GetNameOnly', 'GetNameSpaceArray', 
            'GetNameSpaceOnly', 'GetNameSpacePrefix', 'GetNameWithNameSpacePrefix', 'GetNameWithoutNameSpacePrefix', 'GetNextProperty', 'GetObjectFlags', 
            'GetOriginalSystemUnit', 'GetOriginalUpAxis', 'GetReferenceTo', 'GetReferencedBy', 'GetReferencedByCount', 'GetRootDocument', 'GetScene', 
            'GetSelected', 'GetSrcObject', 'GetSrcObjectCount', 'GetSrcProperty', 'GetSrcPropertyCount', 'GetSystemUnit', 'GetTimeMode', 
            'GetTimelineDefaultTimeSpan', 'GetTypeFlags', 'GetTypeName', 'GetUniqueID', 'GetUrl',
            'IsAReferenceTo', 'IsConnectedDstObject', 'IsConnectedDstProperty', 'IsConnectedSrcObject', 'IsConnectedSrcProperty', 'IsReferencedBy', 
            'Localize', 'RemovePrefix', 'RootProperty', 'SetAllObjectFlags', 'SetAmbientColor', 'SetAxisSystem', 'SetDefaultCamera', 'SetDocument',
            'SetInitialName', 'SetName', 'SetNameSpace', 'SetObjectFlags', 'SetOriginalSystemUnit', 'SetOriginalUpAxis', 'SetSelected', 'SetSystemUnit',
            'SetTimeMode', 'SetTimelineDefaultTimeSpan', 'SetUrl', 'StripPrefix', 
            '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', 
            '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__']
        """

        if verbose==True:
            print ("Before time mode:{}".format(lGlobalSettings.GetTimeMode()))


        # lGlobalSettings.SetTimeMode(FbxTime.EMode.eFrames60)


        if verbose==True:
            print ("After time mode:{}".format(lScene.GetGlobalSettings().GetTimeMode()))

        self.destroyAllAnimation()

        lAnimStackName = pkl_filename
        lAnimStack = FbxAnimStack.Create(lScene, lAnimStackName)
        lAnimLayer = FbxAnimLayer.Create(lScene, "Base Layer")
        lAnimStack.AddMember(lAnimLayer)
        lRootNode = lScene.GetRootNode()


        # Print hierarchy starting from root node
        def print_fbx_hierarchy(node, depth=0):
            print("  " * depth + node.GetName())
            for i in range(node.GetChildCount()):
                print_fbx_hierarchy(node.GetChild(i), depth + 1)
        print("!! debug FBX Hierarchy:")
        print_fbx_hierarchy(lRootNode)

        #######################
        names = SmplObjects.joints
        print('debug names', names)
        names = ['m_avg_'+name for name in names]

        # 1. Write smpl_poses
        smpl_poses = smpl_params["smpl_poses"] ## (num_frames, 72), which are 24 joints * 3
        for idx, name in enumerate(names):
            node = lRootNode.FindChild(name)
            print('!! debug dx, name',idx, name)
            print('         node, ', type(node),node)
            rotvec = smpl_poses[:, idx*3:idx*3+3]
            _euler = []
            for _f in range(rotvec.shape[0]):
                r = R.from_rotvec([rotvec[_f, 0], rotvec[_f, 1], rotvec[_f, 2]])
                euler = r.as_euler('xyz', degrees=True)
                _euler.append([euler[0], euler[1], euler[2]])
            euler = np.vstack(_euler)

            lCurve = node.LclRotation.GetCurve(lAnimLayer, "X", True)
            if lCurve:
                self._write_curve(lCurve, euler[:, 0])
            else:
                print ("Failed to write {}, {}".format(name, "x"))

            lCurve = node.LclRotation.GetCurve(lAnimLayer, "Y", True)
            if lCurve:
                self._write_curve(lCurve, euler[:, 1])
            else:
                print ("Failed to write {}, {}".format(name, "y"))

            lCurve = node.LclRotation.GetCurve(lAnimLayer, "Z", True)
            if lCurve:
                self._write_curve(lCurve, euler[:, 2])
            else:
                print ("Failed to write {}, {}".format(name, "z"))

        # 3. Write smpl_trans to f_avg_root
        smpl_trans = smpl_params["smpl_trans"]
        name = "m_avg_root" #"root"
        node = lRootNode.FindChild(name)
        lCurve = node.LclTranslation.GetCurve(lAnimLayer, "X", True)
        if lCurve:
            self._write_curve(lCurve, smpl_trans[:, 2])
        else:
            print ("Failed to write {}, {}".format(name, "x"))

        lCurve = node.LclTranslation.GetCurve(lAnimLayer, "Y", True) # Translation on the Y axis (in blender, this is not the vertical axis but one of the axis that forms the horizontal plane)
        if lCurve:
            self._write_curve(lCurve, smpl_trans[:, 0])
        else:
            print ("Failed to write {}, {}".format(name, "y"))

        lCurve = node.LclTranslation.GetCurve(lAnimLayer, "Z", True)
        if lCurve:
            self._write_curve(lCurve, smpl_trans[:, 1])
        else:
            print ("Failed to write {}, {}".format(name, "z"))

    def writeFbx(self, write_base:str, filename:str):
        if os.path.isdir(write_base) == False:
            os.makedirs(write_base, exist_ok=True)
        write_path = os.path.join(write_base, filename.replace(".pkl", ""))
        print ("Writing to {}".format(write_path))
        lResult = SaveScene(self.lSdkManager, self.lScene, write_path)

        if lResult == False:
            raise Exception("Failed to write to {}".format(write_path))

    def destroy(self):
        self.lSdkManager.Destroy()

    def destroyAllAnimation(self):
        lScene = self.lScene
        animStackCount = lScene.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimStack.ClassId))
        for i in range(animStackCount):
            lAnimStack = lScene.GetSrcObject(FbxCriteria.ObjectType(FbxAnimStack.ClassId), i)
            lAnimStack.Destroy()
