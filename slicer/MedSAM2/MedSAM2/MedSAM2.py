import logging
import os
from typing import Annotated, Optional
import traceback


import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode

import numpy as np
import tempfile
import threading
import requests
import time

import gzip
import io
import os
import nibabel as nib
import SimpleITK as sitk
import glob

import re

from totalsegmentator.python_api import totalsegmentator
#
# MedSAM2
#


class MedSAM2(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("MedSAM2")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Reza Asakereh (University Health Network)", "Sumin Kim (University of Toronto)", "Jun Ma (University Health Network)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#MedSAM2">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete



#
# MedSAM2ParameterNode
#


@parameterNodeWrapper
class MedSAM2ParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# MedSAM2Widget
#


class MedSAM2Widget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/MedSAM2.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = MedSAM2Logic()
        self.logic.widget = self

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Preprocessing
        self.ui.cmbPrepOptions.addItems(['Manual', 'Abdominal CT', 'Lung CT', 'Brain CT', 'Mediastinum CT', 'MR'])
        self.ui.cmbPrepOptions.currentTextChanged.connect(lambda new_text: self.setManualPreprocessVis(new_text == 'Manual'))
        self.ui.pbApplyPrep.connect('clicked(bool)', lambda: self.logic.applyPreprocess(self.ui.cmbPrepOptions.currentText, self.ui.sldWinLevel.value, self.ui.sldWinWidth.value))
        self.ui.cmbSlicerIdx.addItems(['Select ROI on the current frame'])



        self.checkpoint_list = {
            'Latest': 'MedSAM2_latest.pt',
            'Lesions CT scan': 'MedSAM2_CTLesion.pt',
            'Liver lesions MRI': 'MedSAM2_MRI_LiverLesion.pt',
            'Heart ultra sound': 'MedSAM2_US_Heart.pt',
            'Base model': 'MedSAM2_2411.pt'
        }
        self.ui.cmbCheckpoint.addItems(list(self.checkpoint_list.keys()))
        self.ui.pathModel.connect('currentPathChanged(const QString&)', lambda: setattr(self.logic, 'newModelUploaded', False))
        self.ui.pathConfig.connect('currentPathChanged(const QString&)', lambda: setattr(self.logic, 'newConfigUploaded', False))
        
        # Setting icons
        # Icons used here are downloaded from flaticon's free icons package. Detailed attributes can be found in slicer/MedSAM2/MedSAM2/Resources/Icons/attribute.html 
        from PythonQt.QtGui import QIcon
        iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')
        self.ui.pbApplyPrep.setIcon(QIcon(os.path.join(iconsPath, 'verify.png')))
        self.ui.btnStart.setText("Normalization")
        self.ui.btnROI.setText("Initial Segmentation")
        self.ui.btnSegment.setText("Refined Mask Propagation")
        
        self.ui.btnRefine.setIcon(QIcon(os.path.join(iconsPath, 'performance.png')))
        self.ui.btnSegment.setIcon(QIcon(os.path.join(iconsPath, 'body-scan.png')))
        self.ui.btnRefine3D.setIcon(QIcon(os.path.join(iconsPath, 'performance.png')))
        self.ui.btnAddPoint.setIcon(QIcon(os.path.join(iconsPath, 'add-selection.png')))
        self.ui.btnSubtractPoint.setIcon(QIcon(os.path.join(iconsPath, 'sub-selection.png')))
        self.ui.btnMiddleSlice.setText('show gt mask or turn off')

        # Buttons
        self.ui.btnROI.connect("clicked(bool)", self.logic.run_TotalSegmentator) # 정규화버튼
        self.ui.btnStart.connect("clicked(bool)", self.logic.norm_btn)
        self.ui.btnRefine.connect("clicked(bool)", self.logic.refineMiddleMask)
        self.ui.btnSegment.connect("clicked(bool)", self.logic.segment)
        self.ui.btnRefine3D.connect("clicked(bool)", self.logic.refineMiddleMask)
        self.ui.btnAddPoint.connect("clicked(bool)", lambda: self.addPoint(prefix='addition'))
        self.ui.btnSubtractPoint.connect("clicked(bool)", lambda: self.addPoint(prefix='subtraction'))
        self.ui.btnMiddleSlice.connect("clicked(bool)", lambda: self.logic.showGT(prefix='Mask_spleen2')) # 이거 GT데이터 보여주는 버튼

        self.ui.btnEnd.setVisible(False)
        self.ui.CollapsibleButton_5.setVisible(False)
        self.ui.btnAddPoint.setVisible(False)
        self.ui.btnSubtractPoint.setVisible(False)
        self.ui.btnRefine3D.setVisible(False)
        
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
    
    def setManualPreprocessVis(self, visible):
        self.ui.lblLevel.setVisible(visible)
        self.ui.lblWidth.setVisible(visible)
        self.ui.sldWinLevel.setVisible(visible)
        self.ui.sldWinWidth.setVisible(visible)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[MedSAM2ParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
    
    def addPoint(self, prefix=''):
        planeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', prefix).GetID()
        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        selectionNode.SetReferenceActivePlaceNodeID(planeNode)
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        placeModePersistence = 0
        interactionNode.SetPlaceModePersistence(placeModePersistence)
        # mode 1 is Place, can also be accessed via slicer.vtkMRMLInteractionNode().Place
        interactionNode.SetCurrentInteractionMode(1)

        slicer.mrmlScene.GetNodeByID(planeNode).GetDisplayNode().SetGlyphScale(0.5)
        slicer.mrmlScene.GetNodeByID(planeNode).GetDisplayNode().SetInteractionHandleScale(1)



class MedSAM2Logic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
    norm_b = False
    image_data_hu = None
    image_data_norm = None
    boundaries = None
    volume_node = None
    image_data = None
    widget = None
    middleMaskNode = None
    allSegmentsNode = None
    newModelUploaded = False
    newConfigUploaded = False
    cachedBoundaries = None
    lastSegmentLabel = None
    Total_segmentator_mask_path = None
    Medsam2_segmentator_mask_path = None
    medsam2_index = 0

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return MedSAM2ParameterNode(super().getParameterNode())
    
    # 현재 3D slicer에 올라와있는 이미지로 image_data에 저장하는 함수
    def captureImage(self):
        self.volume_node = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')[0]
        if self.volume_node.GetNodeTagName() == 'LabelMapVolume': ### some volumes are loaded as LabelMapVolume instead of ScalarVolume, temporary
            outputvolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", self.volume_node.GetName())
            sef = slicer.modules.volumes.logic().CreateScalarVolumeFromVolume(slicer.mrmlScene, outputvolume, self.volume_node)
            slicer.mrmlScene.RemoveNode(self.volume_node)

            appLogic = slicer.app.applicationLogic()
            selectionNode = appLogic.GetSelectionNode()
            selectionNode.SetActiveVolumeID(sef.GetID())
            appLogic.PropagateVolumeSelection()
            self.volume_node = sef

        self.image_data = slicer.util.arrayFromVolume(self.volume_node)  ################ Only one node?
    
    def norm_btn(self):
        self.norm_b = not self.norm_b 
        if self.norm_b is True:
            print("정규화 된상태!")
        else:
            print("정규화 안된상태!")
        

    def run_on_background(self, target, args, title):
        self.progressbar = slicer.util.createProgressDialog(autoClose=False)
        self.progressbar.minimum = 0
        self.progressbar.maximum = 0
        self.progressbar.setLabelText(title)
        
        job_event = threading.Event()
        paral_thread = threading.Thread(target=target, args=(*args, job_event,))
        paral_thread.start()
        while not job_event.is_set():
            slicer.app.processEvents()
        paral_thread.join()

        self.progressbar.close()
    
    # Total segmentator로 1차분할한 결과에서 padding더해서 경계박스 만들어주는 함수 
    def get_bounding_info_from_labels(self,gst, padding=3):
        """
        Args:
            gst (np.ndarray): (depth, height, width) NumPy
            padding (int)
        Returns:
            tuple: (slice_idx, bboxes, zrange)
                - slice_idx (int): 경계 박스의 Z축 중심 인덱스.
                - bboxes (list): [[x_min, y_min, x_max, y_max]] 형태의 2D 경계 박스 리스트
                - zrange (list): [z_min, z_max] 형태의 Z축 범위 리스트
                0이 아닌 레이블이 없으면 (None, [], [])를 반환
        """
        coords = np.where(gst > 0)
        
        if coords[0].size == 0:
            # 0이 아닌 레이블이 하나도 없는 경우
            print("0이 아닌 레이블을 찾을 수 없습니다.")
            return None, [], []

        z_min, z_max = np.min(coords[0]), np.max(coords[0])
        y_min, y_max = np.min(coords[1]), np.max(coords[1])
        x_min, x_max = np.min(coords[2]), np.max(coords[2])

        depth, height, width = gst.shape

        z_min_padded = max(0, z_min - padding)
        y_min_padded = max(0, y_min - padding)
        x_min_padded = max(0, x_min - padding)
        
        z_max_padded = min(depth - 1, z_max + padding)
        y_max_padded = min(height - 1, y_max + padding)
        x_max_padded = min(width - 1, x_max + padding)

        slice_idx = (z_min_padded + z_max_padded) // 2
        zrange = [int(z_min_padded), int(z_max_padded)]
        
        boxes_3D = np.zeros((depth, 1, 4), dtype=np.int32)
        
        box_2d = [int(x_min_padded), int(y_min_padded), int(x_max_padded), int(y_max_padded)]

        # zrange 범위에 해당하는 슬라이스에만 2D 박스 정보를 채워 넣는거
        for i in range(zrange[0], zrange[1] + 1):
            boxes_3D[i, 0, :] = box_2d
            
        return slice_idx, zrange, boxes_3D

    # Totasegmentator api로 1차분할 실행
    def run_TotalSegmentator(self):
        if self.volume_node is None:
            raise RuntimeError("활성 볼륨이 없습니다. 먼저 볼륨을 로드/선택하세요.")
        
        self.segmentation(filetype='nifti', roi_organs=['spleen'], ip = self.widget.ui.txtIP.text.strip(), port = self.widget.ui.txtPort.text.strip())

    # segmentation_mask를 가지고 3D slicer에 올려서 시각화해주는 코드
    def showSegmentation(self, segmentation_mask,name, improve_previous=False):
        if self.allSegmentsNode is None:
            self.allSegmentsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

        current_seg_group = self.allSegmentsNode
        current_seg_group.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        labels = np.unique(segmentation_mask)[1:] # all labels except background(0)

        for idx, label in enumerate(labels, start=1):
            curr_object = np.zeros_like(segmentation_mask)
            curr_object[segmentation_mask == label] = label
            #new_seg_label = 'segment_'+str(label)+'_'+str(int(time.time()))
            new_seg_label = 'segment_'+'label_'+str(label)+'_'+ name
            segment_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", new_seg_label)
            slicer.util.updateVolumeFromArray(segment_volume, curr_object)

            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(segment_volume, current_seg_group)
            slicer.util.updateSegmentBinaryLabelmapFromArray(curr_object, current_seg_group, segment_volume.GetName(), self.volume_node)

        if improve_previous:
            print('Removing segment:', self.lastSegmentLabel)
            self.allSegmentsNode.GetSegmentation().RemoveSegment(self.lastSegmentLabel)
        
        self.lastSegmentLabel = new_seg_label
        print('self.lastSegmentLabel is updated to', self.lastSegmentLabel)

    # 실제 Totalsegmentator돌리는 코드
    def segmentation(self, filetype, roi_organs, ip, port):
        # 넘파이배열 -> NIFTI파일로 변환해주고 그걸 TOTALSEGMENTATOR에 INPUT으로
        # 참조 노드에서 IJK→RAS, spacing, origin 가져오기
        ijkToRas = vtk.vtkMatrix4x4()
        self.volume_node.GetIJKToRASMatrix(ijkToRas)
        M = slicer.util.arrayFromVTKMatrix(ijkToRas)  # 4x4
        spacing = np.array(self.volume_node.GetSpacing(), float)   # (x,y,z)
        origin_ras = np.array(self.volume_node.GetOrigin(), float)

        # 방향(RAS) 추출: 상단 3x3은 방향*spacing이므로 열 단위 정규화
        R_scaled = M[:3, :3]
        direction_ras = R_scaled / np.linalg.norm(R_scaled, axis=0, keepdims=True)

        # 새 볼륨 노드 만들고 기하 적용
        outNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "tmp")
        outNode.SetIJKToRASMatrix(ijkToRas)
        outNode.SetOrigin(*origin_ras)
        outNode.SetSpacing(*spacing)

        if self.norm_b:
            print("정규화 된 상태로 Totalsegmentator실행중")
            slicer.util.updateVolumeFromArray(outNode, self.image_data_norm)
        else:
            print("정규화 안된 상태로 Totalsegmentator실행중")
            slicer.util.updateVolumeFromArray(outNode, self.image_data_hu)

        # 부모 트랜스폼이 걸려 있다면 하든(harden)해서 좌표를 고정
        if outNode.GetParentTransformNode():
            slicer.vtkSlicerTransformLogic().hardenTransform(outNode)

        # NIfTI로 저장
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"/path_Temp/result_{timestamp}.nii.gz"
        slicer.util.saveNode(outNode, output_path)

        # 데이터 서버로 보내주기
        SERVER_URL = 'http://%s:%s/Totalsegmentator'%(ip, port)

        INPUT_FILE_PATH = output_path 
        # 모든 장기를 원하면 None으로 설정
        ROI_SUBSET = roi_organs

        if not os.path.exists(INPUT_FILE_PATH):
            print(f"오류: 파일이 존재하지 않습니다 - {INPUT_FILE_PATH}")
            return

        files = {
            'file': (os.path.basename(INPUT_FILE_PATH), open(INPUT_FILE_PATH, 'rb'), 'application/octet-stream')
        }
        data = {
            'roi_subset': ROI_SUBSET,
            'output_type': filetype
        }  

        def _print_volume_info(node, title=""):
            try:
                img = node.GetImageData()
                dims = img.GetDimensions() if img else None
                spacing = node.GetSpacing() if hasattr(node, "GetSpacing") else None
                origin  = node.GetOrigin() if hasattr(node, "GetOrigin") else None
                m = vtk.vtkMatrix4x4()
                node.GetIJKToRASDirectionMatrix(m)
                logging.info(f"[{title}] dims={dims}, spacing={spacing}, origin={origin}")
                logging.info(f"[{title}] IJK->RAS:\n" + "\n".join(["  " + " ".join([f"{m.GetElement(r,c): .5f}" for c in range(4)]) for r in range(4)]))
            except Exception as e:
                logging.error(f"_print_volume_info error: {e}")

        def _unique_safe(arr, name="arr"):
            try:
                u = np.unique(arr)
                logging.info(f"{name} unique (first 20) = {u[:20]}")
                return u
            except Exception as e:
                logging.error(f"unique() failed for {name}: {e}")
                return np.array([])

        try:
            logging.getLogger().setLevel(logging.INFO)
            logging.info(f"서버({SERVER_URL})에 분할 요청을 보냅니다...")

            response = requests.post(SERVER_URL, files=files, data=data, timeout=600)
            logging.info(f"HTTP status: {response.status_code}")

            if response.status_code != 200:
                logging.error(f"오류 발생 (Status Code: {response.status_code}) | text={response.text[:500]}")
            else:
                logging.info("서버로부터 NIfTI 데이터를 성공적으로 받았습니다.")

                # --- 압축 해제 ---
                try:
                    payload = gzip.decompress(response.content)
                    logging.info("gzip.decompress 성공")
                except Exception:
                    payload = response.content
                    logging.info("gzip.decompress 불필요 — 원본 사용")

                # --- NIfTI 로드 ---
                nifti_file_in_memory = io.BytesIO(payload)
                file_holder = nib.FileHolder(fileobj=nifti_file_in_memory)
                file_map = {'image': file_holder}
                nifti_image = nib.Nifti1Image.from_file_map(file_map)

                mask_f = nifti_image.get_fdata(caching='unchanged')
                logging.info(f"NIfTI mask raw: dtype={mask_f.dtype}, shape={mask_f.shape}, "
                            f"min={np.nanmin(mask_f)}, max={np.nanmax(mask_f)}")

                # NaN 방지
                if np.isnan(mask_f).any():
                    logging.warning("mask에 NaN 존재 — 0으로 치환")
                    mask_f = np.nan_to_num(mask_f, nan=0)

                # 정수화
                mask_i = np.rint(mask_f).astype(np.int16, copy=False)
                u0 = _unique_safe(mask_i, "mask_i(before relabel)")

                # 참조 볼륨 확인
                if self.volume_node is None:
                    raise RuntimeError("self.volume_node가 None 입니다. 배경 볼륨이 필요합니다.")
                ref_arr = slicer.util.arrayFromVolume(self.volume_node) 
                logging.info(f"ref shape={ref_arr.shape}")
                _print_volume_info(self.volume_node, "REF_VOLUME")

                # 축/shape 맞춤 
                mask_for_slicer = mask_i
                if mask_i.shape == ref_arr.shape:
                    logging.info("마스크 shape == 참조 shape (OK)")
                elif mask_i.shape[::-1] == ref_arr.shape:
                    logging.info("마스크 shape가 역순 — transpose(2,1,0) 적용")
                    mask_for_slicer = np.transpose(mask_i, (2,1,0))
                else:
                    logging.error(f"shape 불일치: mask={mask_i.shape}, ref={ref_arr.shape}")
                    raise RuntimeError("마스크/참조 볼륨 그리드 불일치 — 서버에서 리샘플 필요")

                # 여기서 Totalsegmentator로 한 분할결과 .npz파일로 저장
                slice_idx, zrange, boxes_3D = self.get_bounding_info_from_labels(mask_for_slicer)
                self.create3DBBoxROI(self.volume_node, boxes_3D, zrange)
                self.showSegmentation(mask_for_slicer,'total')
                script_dir = os.path.dirname(os.path.abspath(__file__))
                save_path = os.path.join(script_dir, "TotalSegmentator_mask_result.npz")
                self.Total_segmentator_mask_path = save_path
                # npz 파일로 저장
                np.savez(save_path, segs=mask_for_slicer)
                print(f"TotalSegmentator_mask_result: {self.Total_segmentator_mask_path}")

            #MedSAM2 후보정 실행
            self.segment()


            self.scroll_limit(zrange[0], zrange[1])
            print(f"z_max: {zrange[1]}")

        except Exception as e:
            logging.error("예외 발생:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)))

        slicer.util.updateVolumeFromArray(outNode, self.image_data) 


    # 분할하기 위해서 서버에 데이터 보내는 함수
    def segment_helper(self, img_path, gts_path, result_path, ip, port, job_event):
        config, checkpoint = self.getConfigCheckpoint()

        self.progressbar.setLabelText(' uploading refined middle slice... ')
        upload_url = 'http://%s:%s/upload'%(ip, port)

        with open(gts_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(upload_url, files=files)

        with open(gts_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(upload_url, files=files)

        self.progressbar.setLabelText(' segmenting... ')
        run_script_url = 'http://%s:%s/run_script'%(ip, port)
        
        print('data sent is: ', {
                'checkpoint': checkpoint,
                'input': os.path.basename(img_path),
                'gts': os.path.basename(gts_path),
                'propagate': 'Y',
                'config': config,
            })

        response = requests.post(
            run_script_url,
            data={
                'checkpoint': checkpoint,
                'input': img_path,
                'gts': gts_path,
                'propagate': 'Y',
                'config': config,
            }
        )

        self.progressbar.setLabelText(' downloading results... ')
        download_file_url = 'http://%s:%s/download_file'%(ip, port)

        response = requests.get(download_file_url, data={'output': 'data/video/segs_tiny/%s'%os.path.basename(img_path)})

        with open(result_path, 'wb') as f: #TODO: arbitrary file name
            f.write(response.content)
        
        job_event.set()
    
    def create3DBBoxROI(self, volumeNode, bboxes, zrange):
        """
        volumeNode: CT 볼륨 (vtkMRMLScalarVolumeNode)
        bboxes: [[x_min, y_min, x_max, y_max], ...]
        zrange: [z_min, z_max] (slice index, IJK z축 기준)
        """
        roi_nodes = []
        print(f"bboxes:{bboxes}")

        bbox = np.squeeze(bboxes, axis = 1)
        i_min, j_min, i_max, j_max = bbox[zrange[0]]
        k_min, k_max = zrange

        corners_ijk = np.array([
            [i_min, j_min, k_min],
            [i_max, j_max, k_max]
        ])

        # --- IJK → RAS ---
        ijkToRas = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRas)
        ijkToRas_np = slicer.util.arrayFromVTKMatrix(ijkToRas)

        corners_ras = []
        for corner in corners_ijk:
            pt = np.array([corner[0], corner[1], corner[2], 1.0])
            ras = ijkToRas_np @ pt
            corners_ras.append(ras[:3])
        corners_ras = np.array(corners_ras)

        # 중심, 크기 계산
        center_ras = (corners_ras[0] + corners_ras[1]) / 2.0
        size_ras = np.abs(corners_ras[1] - corners_ras[0])

        # ROI 노드 생성
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", f"BBoxROI")
        roiNode.SetXYZ(*center_ras)
        roiNode.SetRadiusXYZ(*(size_ras / 2.0))

        # Display 속성
        roiNode.GetDisplayNode().SetVisibility(True)    # ROI 전체 표시
        roiNode.GetDisplayNode().SetVisibility3D(True)  # 3D에 표시
        roiNode.GetDisplayNode().SetVisibility2D(True)  # 2D Slice 뷰에도 표시

        roi_nodes.append(roiNode)

        return roi_nodes

    # MedSAM2로 분할하는 함수
    def segment(self):
        self.captureImage()
        result = None
        # segment할 슬라이스인덱스, 경계박스, z범위
        if self.Medsam2_segmentator_mask_path is None:
            with np.load(self.Total_segmentator_mask_path) as data:
                result = data['segs'].copy()
        else:
            with np.load(self.Medsam2_segmentator_mask_path) as data:
                result = data['segs'].copy()

        result = result.astype(np.int16)
        slice_idx, zrange, boxes_3D = self.get_bounding_info_from_labels(result)
        self.create3DBBoxROI(self.volume_node, boxes_3D, zrange)

        temp_result = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            img_path = "%s\img_data.npz"%(tmpdirname,)
            np.savez(img_path, imgs=self.image_data_norm, boxes=boxes_3D, z_range=[*zrange, slice_idx])
            gts_path = '%s\gts.npz'%(tmpdirname,)
            result_path = '%s\\result.npz'%(tmpdirname,)
            
            temp_seg = self.getSegmentationArray(self.allSegmentsNode)

            if not np.array_equal(result, temp_seg):
                diff = (result != temp_seg)
                slice_has_idff = np.any(diff, axis=(1,2))
                selected_slices = temp_seg*slice_has_idff[:,np.newaxis,np.newaxis]
                z_index = 0
                for slice in selected_slices:
                    if np.all(slice == 0):
                        z_index += 1
                    else:
                        z_index += 1
                np.savez_compressed(gts_path, segs=selected_slices)
            else:
                np.savez_compressed(gts_path, segs=result)

            self.run_on_background(self.segment_helper, (img_path, gts_path, result_path, self.widget.ui.txtIP.text.strip(), self.widget.ui.txtPort.text.strip()), 'Segmenting...')

            # loading results
            segmentation_mask = np.load(result_path, allow_pickle=True)['segs']
            self.showSegmentation(segmentation_mask,f'medsam2_{self.medsam2_index}')
            self.medsam2_index += 1
            temp_result = segmentation_mask.copy()
            self.cachedBoundaries = {'bboxes': boxes_3D, 'zrange': zrange}
            self.widget.ui.CollapsibleButton_5.setVisible(True)

        self.Medsam2_segmentator_mask_path = 'Medsam2_segmentator_mask.npz'
        np.savez_compressed(self.Medsam2_segmentator_mask_path, segs=temp_result)

        roiNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsROINode')
        for roiNode in roiNodes:
            slicer.mrmlScene.RemoveNode(roiNode)
        self.boundaries = None
    

    # 현재 3D slicer에 올라와있는 분할된 마스크를 가지고 넘파이배열로 만들어서 return하는 함수
    def getSegmentationArray(self, segmentationNode=None):
        if segmentationNode is None:
            segmentationNode = self.allSegmentsNode

        if segmentationNode is None:
            raise ValueError("No segmentation node available. Run showSegmentation first.")

        segGroup = segmentationNode.GetSegmentation()
        try:
            segmentIds = list(segGroup.GetSegmentIDs())
        except Exception:
            segmentIds = [segGroup.GetSegmentId(i) for i in range(segGroup.GetNumberOfSegments())]

        reference = slicer.util.arrayFromVolume(self.volume_node)
        result = np.zeros(reference.shape, dtype=np.int16)

        for sid in segmentIds:
            segArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, sid)

            # 라벨 번호를 segment 이름에서 파싱(SEGMENT_() 에 나온다고 가정, 실제로 저장도 그렇게 함)
            name = segGroup.GetSegment(sid).GetName()
            label_numbers = [int(s) for s in name.split('_') if s.isdigit()]
            orig_label = label_numbers[0] if label_numbers else 1 

            if np.any(segArray):
                result[segArray > 0] = orig_label

        uniq, cnt = np.unique(result, return_counts=True)
        print("[getSegmentationArray] labels:", {int(u): int(c) for u, c in zip(uniq, cnt)})

        return result

    # SegmentEditor로 UI로 바꾸는 함수
    def refineMiddleMask(self):
        slicer.util.selectModule("SegmentEditor") 

    # CT 이미지전처리하는 함수
    def preprocess_CT(self, win_level=40.0, win_width=400.0):
        self.captureImage()

        self.image_data_hu = np.copy(self.image_data)

        img = np.copy(self.image_data)
        lower_bound, upper_bound = win_level - win_width/2, win_level + win_width/2
        image_data_pre = np.clip(img, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
        image_data_pre = np.uint8(image_data_pre)

        self.volume_node.GetDisplayNode().SetAutoWindowLevel(False)
        self.volume_node.GetDisplayNode().SetWindowLevelMinMax(0, 255)

        self.image_data_norm = image_data_pre
        
        return image_data_pre
    
    # MRI 이미지전처리하는 함수
    def preprocess_MR(self, lower_percent=0.5, upper_percent=99.5):
        self.captureImage()

        self.image_data_hu = np.copy(self.image_data)
        
        img = np.copy(self.image_data)
        lower_bound, upper_bound = np.percentile(img[img > 0], lower_percent), np.percentile(img[img > 0], upper_percent)
        image_data_pre = np.clip(self.img, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
        image_data_pre = np.uint8(image_data_pre)

        self.volume_node.GetDisplayNode().SetAutoWindowLevel(False)
        self.volume_node.GetDisplayNode().SetWindowLevelMinMax(0, 255)

        self.image_data_norm = image_data_pre
        return image_data_pre
    

    def updateImage(self, new_image):
        self.image_data[:,:,:] = new_image
        slicer.util.arrayFromVolumeModified(self.volume_node)
    
    # 전처리 진행하는함수
    def applyPreprocess(self, method, win_level, win_width):
        if method == 'MR':
            prep_img = self.preprocess_MR()
        elif method == 'Manual':
            prep_img = self.preprocess_CT(win_level = win_level, win_width = win_width)
        else:
            conversion = {
                'Abdominal CT': (400.0, 40.0),
                'Lung CT': (1500.0, -600.0),
                'Brain CT': (80.0, 40.0),
                'Mediastinum CT': (350.0, 50.0),
            }
            ww, wl = conversion[method]
            prep_img = self.preprocess_CT(win_level = wl, win_width = ww)

        self.updateImage(prep_img)

        
    # config파일과 checkpint가져오는 함수
    def getConfigCheckpoint(self):
        if self.widget.ui.pathConfig.currentPath == '':
            config = 'MedSAM2_tiny512.yaml'
        else:
            config = 'custom_' + os.path.basename(self.widget.ui.pathConfig.currentPath)
        
        if self.widget.ui.pathModel.currentPath == '':
            checkpoint = self.widget.checkpoint_list[self.widget.ui.cmbCheckpoint.currentText]
        else:
            model_name = os.path.basename(self.widget.ui.pathModel.currentPath).split('.')[0]
            checkpoint = os.path.join(model_name, os.path.basename(self.widget.ui.pathModel.currentPath))
        
        return config, checkpoint
    
    def scroll_limit(self, z_min, z_max):
        # 좌표 변환
        spacing = self.volume_node.GetSpacing()
        origin = self.volume_node.GetOrigin()
        extent = self.volume_node.GetImageData().GetExtent()

        zCoordMin = origin[2] + (extent[4] + z_min) * spacing[2]
        zCoordMax = origin[2] + (extent[4] + z_max) * spacing[2]

        # Red slice node
        redSliceNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceNode()

        # 제한 함수
        def lockRedSliceRange(caller, event):
            offset = redSliceNode.GetSliceOffset()
            if offset < zCoordMin:
                redSliceNode.SetSliceOffset(zCoordMin)
            elif offset > zCoordMax:
                redSliceNode.SetSliceOffset(zCoordMax)

        # 옵저버 연결 (여기서 핵심은 vtk.vtkCommand.ModifiedEvent 사용)
        redSliceNode.AddObserver(vtk.vtkCommand.ModifiedEvent, lockRedSliceRange)

        # 시작을 z_min으로 맞추기
        redSliceNode.SetSliceOffset(zCoordMin)

        print(f"✅ Red 뷰는 이제 {z_min}~{z_max} 슬라이스 범위에서만 스크롤 가능합니다.")
    
    # GT데이터 시각화 관련 함수들
    def showGT(self, folder_name):
        script_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 실행 파일(.py) 경로
        gt_folder = os.path.join(script_dir,folder_name)
        self.showGTFromFolder(gt_folder)
    
    def showGTFromFolder(self, gt_folder: str, reverse: bool = True):
        """
        gt_folder 안의 PNG 스택을 읽어 기준 볼륨(self.volume_node)에 정확히 오버레이합니다.
        - reverse: True면 슬라이스 순서를 뒤집어서 스택 결합
        - keep_labelmap_node: True면 임시 LabelMap 노드를 남겨둠(디버깅 용)
        """
        try:
            import sitkUtils 
        except Exception:
            sitkUtils = None

        try:
            if self.volume_node is None:
                self.captureImage()
            if self.volume_node is None:
                slicer.util.errorDisplay("활성 볼륨을 찾을 수 없습니다. 먼저 CT/MR 볼륨을 불러오세요.", windowTitle="MedSAM2")
                return None
            
            files = glob.glob(os.path.join(gt_folder, "*.png"))
            if not files:
                slicer.util.errorDisplay(f"폴더에 PNG 파일이 없습니다:\n{gt_folder}", windowTitle="MedSAM2")
                return None

            # 파일명 안 숫자 기준 정렬
            def _last_int_in_name(p):
                m = re.findall(r"\d+", os.path.basename(p))
                return int(m[-1]) if m else 10**12  # 숫자 없으면 뒤로
            files = sorted(files, key=_last_int_in_name)
            if reverse:
                files = list(reversed(files))

            # PNG → SITK 스택 (라벨)
            slices = [sitk.ReadImage(f) for f in files]
            # 컬러 PNG면 채널 하나 선택
            if slices[0].GetNumberOfComponentsPerPixel() > 1:
                slices = [sitk.VectorIndexSelectionCast(s, 0) for s in slices]
            # 각 슬라이스를 합쳐서 3차원배열로
            gt_img = sitk.JoinSeries(slices)  # (x,y,z)

            # 이진/다중 레이블 판정
            # 넘파이 배열로 변환하고
            arr_probe = sitk.GetArrayFromImage(gt_img)  # (z,y,x)
            # 라벨파악
            uniq = np.unique(arr_probe)
            is_binary_like = np.all(np.isin(uniq, [0, 1, 255])) or len(uniq) <= 3
            # 이진이면 0,1로 변환
            if is_binary_like:
                gt_img = sitk.Cast(gt_img > 0, sitk.sitkUInt16)   # 0/1
                print("png가 0,255로 되어있는데이터")
            else:
                gt_img = sitk.Cast(gt_img, sitk.sitkUInt16)       # 레이블 유지
            
            # 기준 볼륨 geometry로 정렬
            if sitkUtils is None:
                slicer.util.errorDisplay("sitkUtils 모듈을 불러올 수 없습니다. Slicer에서 실행 중인지 확인하세요.", windowTitle="MedSAM2")
                return None
            
            # volume_node을 SITK이미지로 가져오고
            ref_img = sitkUtils.PullVolumeFromSlicer(self.volume_node)

            # 크기가 같으면 GEOMETRY만 복사
            if list(gt_img.GetSize()) == list(ref_img.GetSize()):
                gt_img.CopyInformation(ref_img)
                resampled_gt = gt_img
            # 아니면 크기 보간해줌
            else:
                print("크기가 맞지않아 보간합니다(정확하지 않음)")
                resampled_gt = sitk.Resample(
                    gt_img,
                    ref_img,
                    sitk.Transform(),             # identity
                    sitk.sitkNearestNeighbor,     # 라벨 보간
                    0,
                    sitk.sitkUInt16
                )

            # LabelMap 노드로 푸시 → Segmentation으로 import
            label_name = f"GT_Label_{os.path.basename(gt_folder)}"
            labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", label_name)
            sitkUtils.PushVolumeToSlicer(resampled_gt, labelNode)

            # 기준 볼륨과 동일한 트랜스폼 계층 유지
            parentTx = self.volume_node.GetParentTransformNode()
            if parentTx:
                labelNode.SetAndObserveTransformNodeID(parentTx.GetID())

            if self.allSegmentsNode is None:
                self.allSegmentsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "GT_Segmentation")
            segNode = self.allSegmentsNode
            segNode.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

            if parentTx:
                segNode.SetAndObserveTransformNodeID(parentTx.GetID())

            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelNode, segNode)

            # 오버레이 설정
            if not segNode.GetDisplayNode():
                segNode.CreateDefaultDisplayNodes()
            disp = segNode.GetDisplayNode()
            if disp:
                # 2D 표시
                if hasattr(disp, "SetVisibility2DFill"):
                    disp.SetVisibility2DFill(True)
                if hasattr(disp, "SetVisibility2DOutline"):
                    disp.SetVisibility2DOutline(True)
                if hasattr(disp, "SetOpacity2D"):
                    disp.SetOpacity2D(0.5)

            # 배경에 원본 볼륨 활성화 및 뷰 리셋
            appLogic = slicer.app.applicationLogic()
            selNode = appLogic.GetSelectionNode()
            selNode.SetActiveVolumeID(self.volume_node.GetID())
            appLogic.PropagateVolumeSelection()
            slicer.util.resetSliceViews()

            # 피드백
            nz = int(np.count_nonzero(sitk.GetArrayFromImage(resampled_gt)))
            print(f"✅ GT 시각화 완료\n- 폴더: {gt_folder}\n- 슬라이스: {len(files)}\n- 비영값 픽셀: {nz:,}\n- 다중 레이블: {not is_binary_like}")
            return segNode

        except Exception as e:
            import traceback
            slicer.util.errorDisplay(
                "GT 시각화 중 오류가 발생했습니다:\n" + "".join(traceback.format_exception_only(type(e), e)),
                windowTitle="MedSAM2"
            )
            return None
