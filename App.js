import React, { useState, useEffect } from 'react';
import { SafeAreaView, Text, StyleSheet, View, Button } from 'react-native';
import { Camera } from 'expo-camera';
import Svg, { Circle, Line } from 'react-native-svg';

import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';

// Initiate tensor camera
const TensorCamera = cameraWithTensors(Camera);
const inputTensorWidth = 152;
const inputTensorHeight = 200;
const AUTORENDER = true;

export default function App() {
  const [hasCameraPermission, setHasCameraPermission] = useState(null);
  const [camType, setCamType] = useState(Camera.Constants.Type.front);

  const [isLoading, setIsLoading] = useState(true);
  const [posenetModel, setPosenetModel] = useState(null);
  const [pose, setPose] = useState(null);

  // POSNET MODEL LOADER
  // async function loadPosenetModel() {}

  // Ask for camera permission on mount and set posnet model for later use
  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestPermissionsAsync();
      setHasCameraPermission(status === 'granted');

      await tf.ready();

      const model = await posenet.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: { width: inputTensorWidth, height: inputTensorHeight },
        multiplier: 0.75,
        quantBytes: 2,
      });
      setPosenetModel(model);

      setIsLoading(false);
    })();
  }, []);

  async function handleImageTensorReady(images, updatePreview, gl) {
    const loop = async () => {
      if (!AUTORENDER) {
        updatePreview();
      }

      try {
        if (posenetModel != null) {
          const imageTensor = images.next().value;
          const poseData = await posenetModel.estimateSinglePose(imageTensor, {
            flipHorizontal: false,
          });
          setPose(poseData);
          tf.dispose([imageTensor]);
        }
      } catch (error) {
        // eslint-disable-next-line no-console
        console.log(error);
      }

      if (!AUTORENDER) {
        gl.endFrameEXP();
      }
      // eslint-disable-next-line no-undef
      requestAnimationFrame(loop);
    };
    loop();
  }

  // MAP AND RENDER THE SVG POINTS
  const renderPose = () => {
    const MIN_KEYPOINT_SCORE = 0.7;

    if (pose != null) {
      const keypoints = pose.keypoints
        .filter((k) => k.score > MIN_KEYPOINT_SCORE)
        .map((k, i) => (
          <Circle
            // eslint-disable-next-line react/no-array-index-key
            key={`keypoint${i}`}
            cx={k.position.x}
            cy={k.position.y}
            r="2"
            strokeWidth="0"
            fill="blue"
          />
        ));
      const adjacentKeypoints = posenet.getAdjacentKeyPoints(pose.keypoints, MIN_KEYPOINT_SCORE);

      const skeleton = adjacentKeypoints.map(([from, to], i) => (
        <Line
          // eslint-disable-next-line react/no-array-index-key
          key={`skeletonls_${i}`}
          x1={from.position.x}
          y1={from.position.y}
          x2={to.position.x}
          y2={to.position.y}
          stroke="magenta"
          strokeWidth="1"
        />
      ));

      return (
        <Svg height="100%" width="100%" viewBox={`0 0 ${inputTensorWidth} ${inputTensorHeight}`}>
          {keypoints}
          {skeleton}
        </Svg>
      );
    }
    return null;
  };

  if (hasCameraPermission === null) {
    return <View />;
  }
  if (hasCameraPermission === false) {
    return <Text>No access to camera</Text>;
  }

  return (
    <SafeAreaView>
      <View style={{ width: '100%' }}>
        <View style={styles.sectionContainer}>
          <Button
            onPress={() => {
              if (camType === Camera.Constants.Type.front) {
                setCamType(Camera.Constants.Type.back);
              }
              if (camType === Camera.Constants.Type.back) {
                setCamType(Camera.Constants.Type.front);
              }
            }}
            title="Switch Cam"
          />
          <Text>{isLoading ? 'Tensor Loading' : 'Ready'}</Text>
        </View>
        {!isLoading ? (
          <View style={styles.cameraContainer}>
            <TensorCamera
              // Standard Camera props
              style={styles.camera}
              type={camType}
              zoom={0}
              // Tensor related props
              cameraTextureHeight={1920}
              cameraTextureWidth={1080}
              resizeHeight={inputTensorHeight}
              resizeWidth={inputTensorWidth}
              resizeDepth={3}
              onReady={handleImageTensorReady}
              autorender={AUTORENDER}
            />
            <View style={styles.modelResults}>{renderPose()}</View>
          </View>
        ) : null}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  loadingIndicator: {
    position: 'absolute',
    top: 20,
    right: 20,
    zIndex: 200,
  },
  sectionContainer: {
    marginTop: 32,
    paddingHorizontal: 24,
  },
  camera: {
    position: 'absolute',
    left: 50,
    top: 100,
    width: 600 / 2,
    height: 800 / 2,
    zIndex: 1,
    borderWidth: 1,
    borderColor: 'black',
    borderRadius: 0,
  },
  cameraContainer: {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    height: '100%',
    backgroundColor: '#fff',
  },
  modelResults: {
    position: 'absolute',
    left: 50,
    top: 100,
    width: 600 / 2,
    height: 800 / 2,
    zIndex: 20,
    borderWidth: 1,
    borderColor: 'black',
    borderRadius: 0,
  },
});
