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
        self.ui.btnStart.setText("Use Current Slice")
        self.ui.btnEnd.setText("Total Segmentator")
        def _fix_current_slice_idx():
            try:
                k = self.logic.getCurrentSliceKIndex("Red")
                self.logic.slice_idx_override = int(k)
                print(f"[MedSAM2] slice_idx_override set to k={k}")
                slicer.util.infoDisplay(f"Current slice fixed: k={k}", windowTitle="MedSAM2")
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to fix current slice: {e}", windowTitle="MedSAM2")
        
        self.ui.btnStart.connect("clicked()", _fix_current_slice_idx)
        self.ui.btnRefine.setIcon(QIcon(os.path.join(iconsPath, 'performance.png')))
        self.ui.btnSegment.setIcon(QIcon(os.path.join(iconsPath, 'body-scan.png')))
        self.ui.btnRefine3D.setIcon(QIcon(os.path.join(iconsPath, 'performance.png')))
        self.ui.btnAddPoint.setIcon(QIcon(os.path.join(iconsPath, 'add-selection.png')))
        self.ui.btnSubtractPoint.setIcon(QIcon(os.path.join(iconsPath, 'sub-selection.png')))
        self.ui.btnImprove.setIcon(QIcon(os.path.join(iconsPath, 'continuous-improvement.png')))

        # Buttons
        self.ui.btnEnd.connect("clicked(bool)", self.logic.run_TotalSegmentator)
        self.ui.btnRefine.connect("clicked(bool)", self.logic.refineMiddleMask)
        self.ui.btnSegment.connect("clicked(bool)", self.logic.segment)
        self.ui.btnRefine3D.connect("clicked(bool)", self.logic.refineMiddleMask)
        self.ui.btnAddPoint.connect("clicked(bool)", lambda: self.addPoint(prefix='addition'))
        self.ui.btnSubtractPoint.connect("clicked(bool)", lambda: self.addPoint(prefix='subtraction'))
        self.ui.btnImprove.connect("clicked(bool)", lambda: self.logic.improveResult())

        self.ui.btnROI.setVisible(False)
        self.ui.btnMiddleSlice.setVisible(False)
        self.ui.CollapsibleButton_5.setVisible(False)
        self.ui.btnAddPoint.setVisible(False)
        self.ui.btnSubtractPoint.setVisible(False)
        self.ui.btnImprove.setVisible(False)

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



#
# MedSAM2Logic
#


class MedSAM2Logic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

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

    ##custom 변수
    slice_idx_override = None

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return MedSAM2ParameterNode(super().getParameterNode())
    
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

    
    def get_point_coords(self):
        self.captureImage()
        pointNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode')

        # If volume node is transformed, apply that transform to get volume's RAS coordinates
        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, self.volume_node.GetParentTransformNode(), transformRasToVolumeRas)
        
        point_list = {}
        for pointNode in pointNodes:
            bounds = np.zeros(6)
            pointNode.GetBounds(bounds)
            curr_point = bounds[::2].copy()
            ijk_points = []

            # Get point coordinate in RAS
            point_VolumeRas = transformRasToVolumeRas.TransformPoint(curr_point)

            # Get voxel coordinates from physical coordinates
            volumeRasToIjk = vtk.vtkMatrix4x4()
            self.volume_node.GetRASToIJKMatrix(volumeRasToIjk)
            point_Ijk = [0, 0, 0, 1]
            volumeRasToIjk.MultiplyPoint(np.append(point_VolumeRas,1.0), point_Ijk)
            point_Ijk = [ int(round(c)) for c in point_Ijk[0:3] ]
            
            point_list[pointNode.GetID()] = point_Ijk

        return point_list
    
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
    
    
    def get_bounding_info_from_labels(self,gst, padding=10):
        """
        Args:
            gst (np.ndarray): 정수 레이블을 포함하는 (depth, height, width) 형태의 3D NumPy 배열.
            padding (int): 경계 박스에 추가할 여백(픽셀 단위).

        Returns:
            tuple: (slice_idx, bboxes, zrange) 형태의 튜플.
                - slice_idx (int): 경계 박스의 Z축 중심 인덱스.
                - bboxes (list): [[x_min, y_min, x_max, y_max]] 형태의 2D 경계 박스 리스트.
                - zrange (list): [z_min, z_max] 형태의 Z축 범위 리스트.
                0이 아닌 레이블이 없으면 (None, [], [])를 반환합니다.
        """
        # 0이 아닌 모든 요소의 좌표를 찾습니다.
        # np.where는 각 차원에 대한 인덱스 배열들의 튜플을 반환합니다. (z_indices, y_indices, x_indices)

        coords = np.where(gst > 0)
        
        if coords[0].size == 0:
            # 0이 아닌 레이블이 하나도 없는 경우
            print("경고: 0이 아닌 레이블을 찾을 수 없습니다.")
            return None, [], []

        # 각 차원(z, y, x)의 최소값과 최대값을 찾습니다.
        z_min, z_max = np.min(coords[0]), np.max(coords[0])
        y_min, y_max = np.min(coords[1]), np.max(coords[1])
        x_min, x_max = np.min(coords[2]), np.max(coords[2])

        # 경계 검사를 위해 입력 배열의 shape을 가져옵니다.
        depth, height, width = gst.shape

        # 전체 shape기준으로 원본 이미지를 경계박스가 벗어나지 않게끔
        z_min_padded = max(0, z_min - padding)
        y_min_padded = max(0, y_min - padding)
        x_min_padded = max(0, x_min - padding)
        
        z_max_padded = min(depth - 1, z_max + padding)
        y_max_padded = min(height - 1, y_max + padding)
        x_max_padded = min(width - 1, x_max + padding)

        # 반환할 값들을 계산합니다.
        slice_idx = (z_min_padded + z_max_padded) // 2
        zrange = [int(z_min_padded), int(z_max_padded)]
        
        # (D, 1, 4) 형태의 최종 3D 박스 배열을 생성합니다.
        # 우선 모든 값을 0으로 초기화합니다.
        boxes_3D = np.zeros((depth, 1, 4), dtype=np.int32)
        
        # 2D 박스 정보를 만듭니다.
        box_2d = [
            int(x_min_padded), 
            int(y_min_padded), 
            int(x_max_padded), 
            int(y_max_padded)
        ]

        # zrange 범위에 해당하는 슬라이스에만 2D 박스 정보를 채워 넣습니다.
        for i in range(zrange[0], zrange[1] + 1):
            boxes_3D[i, 0, :] = box_2d
            
        return slice_idx, zrange, boxes_3D


    def run_TotalSegmentator(self):
        if self.volume_node is None:
            raise RuntimeError("활성 볼륨이 없습니다. 먼저 볼륨을 로드/선택하세요.")
        
        self.segmentation(filetype='nifti', roi_organs=['spleen','liver'], ip = self.widget.ui.txtIP.text.strip(), port = self.widget.ui.txtPort.text.strip())

    def showSegmentation(self, segmentation_mask, improve_previous=False):
        if self.allSegmentsNode is None:
            self.allSegmentsNode = slicer.mrmlScene.AddNewNoshowSegmentationdeByClass("vtkMRMLSegmentationNode")

        current_seg_group = self.allSegmentsNode
        current_seg_group.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        labels = np.unique(segmentation_mask)[1:] # all labels except background(0)
        print(f"np.unique(segmentation_mask): {np.unique(segmentation_mask)}")
        print(f"np.unique(segmentation_mask)[1:: {np.unique(segmentation_mask)[1:]}")

        print(f"labels: {labels}")

        for idx, label in enumerate(labels, start=1):
            print(f"label!!! {label}")
            curr_object = np.zeros_like(segmentation_mask)
            curr_object[segmentation_mask == label] = label
            new_seg_label = 'segment_'+str(label)+'_'+str(int(time.time()))
            segment_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", new_seg_label)
            slicer.util.updateVolumeFromArray(segment_volume, curr_object)

            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(segment_volume, current_seg_group)
            slicer.util.updateSegmentBinaryLabelmapFromArray(curr_object, current_seg_group, segment_volume.GetName(), self.volume_node)

        if improve_previous:
            print('Removing segment:', self.lastSegmentLabel)
            self.allSegmentsNode.GetSegmentation().RemoveSegment(self.lastSegmentLabel)
        
        self.lastSegmentLabel = new_seg_label
        print('self.lastSegmentLabel is updated to', self.lastSegmentLabel)

    def segmentation(self, filetype, roi_organs, ip, port):
        print("Min:", self.image_data_hu .min(), "Max:", self.image_data_hu .max())
        print("Min:", self.image_data_norm .min(), "Max:", self.image_data_norm .max())
        # 일단 여기서 넘파이배열 -> NIFTI파일로 변환해주고 그걸 TOTALSEGMENTATOR에 넣어주기
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

        # 넘파이 배열 넣기 (Slicer 기대 형상: KJI = (slices, rows, cols) = (Z,Y,X))
        #이거 totalsegmentator에 넣는 이미지
        slicer.util.updateVolumeFromArray(outNode, self.image_data_norm)

        # 부모 트랜스폼이 걸려 있다면 하든(harden)해서 좌표를 고정
        if outNode.GetParentTransformNode():
            slicer.vtkSlicerTransformLogic().hardenTransform(outNode)

        # NIfTI로 저장
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"/path_Temp/result_{timestamp}.nii.gz"
        slicer.util.saveNode(outNode, output_path)

        print("넘파이배열 2 nifti 저장완료!!!!!!!")

        # Flask 서버의 주소와 포트를 지정합니다.

        SERVER_URL = 'http://%s:%s/Totalsegmentator'%(ip, port)

        # 분할을 요청할 의료 영상 파일 경로
        INPUT_FILE_PATH = output_path # 👈 실제 파일 경로로 변경하세요.

        # 다운로드 받을 결과 파일의 이름
        OUTPUT_FILE_PATH = 'result.nii.gz'

        # 분할을 원하는 장기 목록 (콤마로 구분된 문자열)
        # 예: "lung,liver,kidney,spleen"
        # 모든 장기를 원하면 None으로 설정
        ROI_SUBSET = roi_organs

        # 원하는 출력 파일 타입
        """서버에 분할 요청을 보내고 결과를 저장하는 함수"""
        if not os.path.exists(INPUT_FILE_PATH):
            print(f"오류: 파일이 존재하지 않습니다 - {INPUT_FILE_PATH}")
            return

        # 요청에 포함할 데이터 (파일과 파라미터)
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
                ref_arr = slicer.util.arrayFromVolume(self.volume_node)  # Slicer shape: (k, j, i)
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
                self.showSegmentation(mask_for_slicer)
                script_dir = os.path.dirname(os.path.abspath(__file__))
                save_path = os.path.join(script_dir, "TotalSegmentator_mask_result.npz")
                self.Total_segmentator_mask_path = save_path
                # npz 파일로 저장
                np.savez(save_path, segs=mask_for_slicer)
                print(f"여기서 시작이야(분할한거)! TotalSegmentator_mask_result: {self.Total_segmentator_mask_path}")

        except Exception as e:
            logging.error("예외 발생:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)))

        slicer.util.updateVolumeFromArray(outNode, self.image_data) # 정규화한 값으로 3D slicer Node주기


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


    def segment(self):
        self.captureImage()
        # segment할 슬라이스인덱스, 경계박스, z범위
        data = np.load(self.Total_segmentator_mask_path)
        result = data['segs']
        result = result.astype(np.int16)
        print("Total로 분할된 결과 크기 및 라벨 unique:")
        print(f"result shape: {result.shape}")
        print(f"result shape: {np.unique(result)}")
        slice_idx, zrange, boxes_3D = self.get_bounding_info_from_labels(result)
        print(f"bboxes: {boxes_3D}")
        print(f"zrange: {zrange}")
        print(f"slice_idx: {slice_idx}")



        with tempfile.TemporaryDirectory() as tmpdirname:
            print("Min:", self.image_data_norm.min(), "Max:", self.image_data_norm.max())
            img_path = "%s\img_data.npz"%(tmpdirname,)
            np.savez(img_path, imgs=self.image_data_norm, boxes=boxes_3D, z_range=[*zrange, slice_idx])
            gts_path = '%s\gts.npz'%(tmpdirname,)
            result_path = '%s\\result.npz'%(tmpdirname,)
            temp_seg = self.getSegmentationArray(self.allSegmentsNode)
            if not np.array_equal(result, temp_seg):
                print("A와 C는 하나라도 요소가 다릅니다.")
            else:
                print("A와 C는 완전히 같습니다.")
            np.savez(gts_path, segs=temp_seg)
            if self.Total_segmentator_mask_path:
                np.savez(self.Total_segmentator_mask_path, segs=temp_seg)
            self.run_on_background(self.segment_helper, (img_path, gts_path, result_path, self.widget.ui.txtIP.text.strip(), self.widget.ui.txtPort.text.strip()), 'Segmenting...')

            # loading results
            segmentation_mask = np.load(result_path, allow_pickle=True)['segs']
            print(f"final_segmentation_mask: {np.unique(segmentation_mask)}")
            print(f"result_path!!: {result_path}")
            self.showSegmentation(segmentation_mask)

            # caching box info for possible "segmentation improvement"
            self.cachedBoundaries = {'bboxes': boxes_3D, 'zrange': zrange}

            self.widget.ui.CollapsibleButton_5.setVisible(True)

        roiNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsROINode')
        for roiNode in roiNodes:
            slicer.mrmlScene.RemoveNode(roiNode)
        self.boundaries = None
    
    def middle_mask_helper(self, img_path, result_path, ip, port, job_event):
        config, checkpoint = self.getConfigCheckpoint()
        
        # ui에 모델 파일경로가 있을경우
        if self.widget.ui.pathModel.currentPath != '' and not self.newModelUploaded:
            # TODO: Check if model is valid
            self.progressbar.setLabelText(' uploading model... ')
            upload_url = 'http://%s:%s/upload_model'%(ip, port)

            with open(self.widget.ui.pathModel.currentPath, 'rb') as file:
                files = {'file': file}
                response = requests.post(upload_url, files=files)
                self.newModelUploaded = True # used for caching
        
        # 딥러닝 config파일 서버에 업로드
        if self.widget.ui.pathConfig.currentPath != '' and not self.newConfigUploaded:
            # TODO: Check if config is valid
            self.progressbar.setLabelText(' uploading config file... ')
            upload_url = 'http://%s:%s/upload_config'%(ip, port)

            with open(self.widget.ui.pathConfig.currentPath, 'rb') as file:
                files = {'file': file}
                response = requests.post(upload_url, files=files)
                self.newConfigUploaded = True # used for caching

        # 분할할 이미지데이터를 서버에 업로드
        self.progressbar.setLabelText(' uploading image... ')
        upload_url = 'http://%s:%s/upload'%(ip, port)

        with open(img_path, 'rb') as file:
            # img_path에 원본 이미지의 넘파이배열과 경계박스 그리고 z축범위가 들어가 가있음.
            files = {'file': file}
            response = requests.post(upload_url, files=files)


        # 서버인 run_script에 데이터 주면서 결과추론
        self.progressbar.setLabelText(' segmenting... ')
        run_script_url = 'http://%s:%s/run_script'%(ip, port)

        print('data sent is: ', {
                'checkpoint': checkpoint,
                'input': os.path.basename(img_path),
                'gts': 'X',
                'propagate': 'N',
                'config': config,
            })

        response = requests.post(
            run_script_url,
            data={
                'checkpoint': checkpoint,
                'input': os.path.basename(img_path),
                'gts': 'X',
                'propagate': 'N',
                'config': config,
            }
        )

        # 분할결과를 서버에서 다운로드
        self.progressbar.setLabelText(' downloading results... ')
        download_file_url = 'http://%s:%s/download_file'%(ip, port)

        response = requests.get(download_file_url, data={'output': 'data/video/segs_tiny/%s'%os.path.basename(img_path)})

        with open(result_path, 'wb') as f:
            f.write(response.content)
        
        job_event.set()

    def getSegmentationArray(self, segmentationNode=None):
        """
        Convert SegmentationNode into 3D numpy array.
        Preserves original TotalSegmentator label values (no reindexing).
        """
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

            # ⚡ 핵심: 여기서 라벨 번호를 segment 이름에서 파싱
            name = segGroup.GetSegment(sid).GetName()
            label_numbers = [int(s) for s in name.split('_') if s.isdigit()]
            orig_label = label_numbers[0] if label_numbers else 1 

            if np.any(segArray):
                result[segArray > 0] = orig_label

        uniq, cnt = np.unique(result, return_counts=True)
        print("[getSegmentationArray] labels:", {int(u): int(c) for u, c in zip(uniq, cnt)})

        return result

    
    def refineMiddleMask(self):
        slicer.util.selectModule("SegmentEditor") 

    
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
    

    def improve_helper(self, img_path, result_path, ip, port, job_event):
        self.progressbar.setLabelText(' uploading improvement details... ')
        upload_url = 'http://%s:%s/upload'%(ip, port)

        with open(img_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(upload_url, files=files)

        self.progressbar.setLabelText(' improving... ')
        improve_url = 'http://%s:%s/improve'%(ip, port)

        print('data sent is: ', {
                'input': os.path.basename(img_path),
            })

        response = requests.post(
            improve_url,
            data={
                'input': os.path.basename(img_path),
            }
        )


        self.progressbar.setLabelText(' downloading results... ')
        download_file_url = 'http://%s:%s/download_file'%(ip, port)

        response = requests.get(download_file_url, data={'output': 'data/video/segs_tiny/%s'%os.path.basename(img_path)})

        with open(result_path, 'wb') as f:
            f.write(response.content)
        
        job_event.set()
    

    def improveResult(self):
        # TODO: Make sure a full inference is already performed [you can use self.cachedBoundaries]
        # TODO: Make sure that new points fall within the latest inference bounds

        point_list = self.get_point_coords()
        points_partition = {'addition': [], 'subtraction': []}
        for point_name in point_list:
            point_type = 'addition' if 'addition' in slicer.util.getNode(point_name).GetName() else 'subtraction'
            points_partition[point_type].append(point_list[point_name])
        print(points_partition)

        with tempfile.TemporaryDirectory() as tmpdirname:
            img_path = "%s/img_data.npz"%(tmpdirname,)
            result_path = "%s/result.npz"%(tmpdirname,)
            np.savez(img_path,
                bboxes=self.cachedBoundaries['bboxes'],
                zrange=self.cachedBoundaries['zrange'],
                points_addition=points_partition['addition'],
                points_subtraction=points_partition['subtraction'],
                img_size=self.image_data.shape[:3]
            )
            self.run_on_background(self.improve_helper, (img_path, result_path, self.widget.ui.txtIP.text.strip(), self.widget.ui.txtPort.text.strip()), 'Improving Segmentation...')
            
            # loading results
            segmentation_mask = np.load(result_path, allow_pickle=True)['segs']
            self.showSegmentation(segmentation_mask, improve_previous=True)
        
        pointNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode')
        for pointNode in pointNodes:
            slicer.mrmlScene.RemoveNode(pointNode)

# custom함수
    def getCurrentSliceKIndex(self, viewName="Red"):
        # 1) 필요한 노드/행렬들
        sliceLogic = slicer.app.layoutManager().sliceWidget(viewName).sliceLogic()
        sliceNode = sliceLogic.GetSliceNode()
        xyToRas = vtk.vtkMatrix4x4()
        xyToRas.DeepCopy(sliceNode.GetXYToRAS())

        # XY 원점(화면 중앙)이 slice 평면의 기준점
        ras = [0.0, 0.0, 0.0, 1.0]
        ras = list(xyToRas.MultiplyPoint(ras))  # [x,y,z,1]

        # 2) (월드)RAS → (볼륨)RAS
        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            None, self.volume_node.GetParentTransformNode(), transformRasToVolumeRas
        )
        rasVol = transformRasToVolumeRas.TransformPoint(ras[:3])

        # 3) (볼륨)RAS → IJK
        rasToIJK = vtk.vtkMatrix4x4()
        self.volume_node.GetRASToIJKMatrix(rasToIJK)
        ijkH = [rasVol[0], rasVol[1], rasVol[2], 1.0]
        out = [0.0, 0.0, 0.0, 0.0]
        rasToIJK.MultiplyPoint(ijkH, out)
        k = int(round(out[2]))

        # 안전 클램핑
        k = max(0, min(k, self.image_data.shape[0] - 1))
        return k

#
# MedSAM2Test
#


class MedSAM2Test(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_MedSAM21()

    def test_MedSAM21(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        self.delayDisplay("Test passed")
