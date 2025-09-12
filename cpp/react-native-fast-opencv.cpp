#include <iostream>
#include <android/log.h>

#include "react-native-fast-opencv.h"
#include "jsi/Promise.h"
#include "jsi/TypedArray.h"
#include <FOCV_Ids.hpp>
#include <FOCV_Storage.hpp>
#include <FOCV_Function.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include "FOCV_Object.hpp"
#include "ConvertImage.hpp"
#include "FOCV_JsiObject.hpp"
#include "jsi/jsi.h"
#include "opencv2/opencv.hpp"

using namespace mrousavy;

void OpenCVPlugin::installOpenCV(jsi::Runtime& runtime, std::shared_ptr<react::CallInvoker> callInvoker) {

    auto func = [=](jsi::Runtime& runtime,
                        const jsi::Value& thisArg,
                        const jsi::Value* args,
                        size_t count) -> jsi::Value {
        auto plugin = std::make_shared<OpenCVPlugin>(callInvoker);
        auto result = jsi::Object::createFromHostObject(runtime, plugin);

        return result;
    };

    auto jsiFunc = jsi::Function::createFromHostFunction(runtime,
        jsi::PropNameID::forUtf8(runtime, "__loadOpenCV"),
        1,
        func);

    runtime.global().setProperty(runtime, "__loadOpenCV", jsiFunc);

}

OpenCVPlugin::OpenCVPlugin(std::shared_ptr<react::CallInvoker> callInvoker) : _callInvoker(callInvoker) {}

jsi::Value OpenCVPlugin::get(jsi::Runtime& runtime, const jsi::PropNameID& propNameId) {
  auto propName = propNameId.utf8(runtime);

  if (propName == "frameBufferToMat") {
    return jsi::Function::createFromHostFunction(
        runtime, jsi::PropNameID::forAscii(runtime, "frameBufferToMat"), 5,
        [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
            size_t count) -> jsi::Object {
        auto rows = arguments[0].asNumber();
        auto cols = arguments[1].asNumber();
        auto channels = arguments[2].asNumber();
        auto input = arguments[3].asObject(runtime);

        auto type = -1;
        if (channels == 1) {
          type = CV_8U;
        }
        if (channels == 3) {
          type = CV_8UC3;
        }
        if (channels == 4) {
          type = CV_8UC4;
        }

        if (channels == -1) {
          throw std::runtime_error("Fast OpenCV Error: Invalid channel count passed to frameBufferToMat!");
        }

        auto inputBuffer = getTypedArray(runtime, std::move(input));
        auto vec = inputBuffer.toVector(runtime);

        cv::Mat mat(rows, cols, type);
        memcpy(mat.data, vec.data(), (int)rows * (int)cols * (int)channels);
        auto id = FOCV_Storage::save(mat);

        return FOCV_JsiObject::wrap(runtime, "mat", id);
    });
  } else  if (propName == "bufferToMat") {
    return jsi::Function::createFromHostFunction(
        runtime, jsi::PropNameID::forAscii(runtime, "bufferToMat"), 5,
        [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
            size_t count) -> jsi::Object {
        try {

        auto type = arguments[0].asString(runtime).utf8(runtime);
        auto rows = arguments[1].asNumber();
        auto cols = arguments[2].asNumber();
        auto channels = arguments[3].asNumber();
        auto input = arguments[4].asObject(runtime);

        auto modeType = -1;
        auto typeSize = 1;

        if(type == "uint8") {
          typeSize = 1;
          if (channels == 1) modeType = CV_8U;
          if (channels == 3) modeType = CV_8UC3;
          if (channels == 4) modeType = CV_8UC4;
        } else if(type == "uint16") {
          typeSize = 2;
          if (channels == 1) modeType = CV_16U;
          if (channels == 3) modeType = CV_16UC3;
          if (channels == 4) modeType = CV_16UC4;
        } else if(type == "int8") {
          typeSize = 1;
          if (channels == 1) modeType = CV_8S;
          if (channels == 3) modeType = CV_8SC3;
          if (channels == 4) modeType = CV_8SC4;
        } else if(type == "int16") {
          typeSize = 2;
          if (channels == 1) modeType = CV_16S;
          if (channels == 3) modeType = CV_16SC3;
          if (channels == 4) modeType = CV_16SC4;
        } else if(type == "int32") {
          typeSize = 4;
          if (channels == 1) modeType = CV_32S;
          if (channels == 3) modeType = CV_32SC3;
          if (channels == 4) modeType = CV_32SC4;
        } else if(type == "float32") {
          typeSize = 4;
          if (channels == 1) modeType = CV_32F;
          if (channels == 3) modeType = CV_32FC3;
          if (channels == 4) modeType = CV_32FC4;
        } else if(type == "float64") {
          typeSize = 8;
          if (channels == 1) modeType = CV_64F;
          if (channels == 3) modeType = CV_64FC3;
          if (channels == 4) modeType = CV_64FC4;
        }

        if (channels == -1) {
          throw std::runtime_error("Fast OpenCV Error: Invalid channel count passed to frameBufferToMat!");
        }

        auto inputBuffer = getTypedArray(runtime, std::move(input));
        auto vec = inputBuffer.toVector(runtime);

        cv::Mat mat(rows, cols, modeType);
        memcpy(mat.data, vec.data(), (int)rows * (int)cols * (int)channels * typeSize);
          __android_log_print(ANDROID_LOG_VERBOSE, "TESTTEST", "%f, %f, %f, %d", rows,cols,channels,typeSize);
        auto id = FOCV_Storage::save(mat);

        return FOCV_JsiObject::wrap(runtime, "mat", id);
        } catch(const std::exception& ex) {

          __android_log_print(ANDROID_LOG_VERBOSE, "TESTTEST", "The error %s", ex.what());
        }
    });
  }
  else if (propName == "base64ToMat") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "base64ToMat"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Object {

          auto base64 = arguments[0].asString(runtime).utf8(runtime);
          __android_log_print(ANDROID_LOG_VERBOSE, "ImageComparison", "%i", base64.length());

          auto mat = ImageConverter::str2mat(base64);
          auto id = FOCV_Storage::save(mat);

          return FOCV_JsiObject::wrap(runtime, "mat", id);
      });
    }
  else if (propName == "matToBuffer") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "matToBuffer"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Object {

                auto id = FOCV_JsiObject::id_from_wrap(runtime, arguments[0]);
                auto mat = *FOCV_Storage::get<cv::Mat>(id);

                jsi::Object value(runtime);

                value.setProperty(runtime, "cols", jsi::Value(mat.cols));
                value.setProperty(runtime, "rows", jsi::Value(mat.rows));
                value.setProperty(runtime, "channels", jsi::Value(mat.channels()));

                auto type = arguments[1].asString(runtime).utf8(runtime);
                auto size = mat.cols * mat.rows * mat.channels();

                if(type == "uint8") {
          __android_log_print(ANDROID_LOG_VERBOSE, "TESTTEST", "size: %d, %d, %d, %d", size, mat.cols, mat.rows, mat.channels());
                  auto arr = TypedArray<TypedArrayKind::Uint8Array>(runtime, size);
          __android_log_print(ANDROID_LOG_VERBOSE, "TESTTEST", "data size: %d", mat.size);
                  arr.updateUnsafe(runtime, (uint8_t*)mat.data, size * sizeof(uint8_t));
                  value.setProperty(runtime, "buffer", arr);
                } else if(type == "uint16") {
                  auto arr = TypedArray<TypedArrayKind::Uint16Array>(runtime, size);
                  arr.updateUnsafe(runtime, (uint16_t*)mat.data, size * sizeof(uint16_t));
                  value.setProperty(runtime, "buffer", arr);
                } else if(type == "uint32") {
                  auto arr = TypedArray<TypedArrayKind::Uint32Array>(runtime, size);
                  arr.updateUnsafe(runtime, (uint32_t*)mat.data, size * sizeof(uint32_t));
                  value.setProperty(runtime, "buffer", arr);
                } else if(type == "int8") {
                  auto arr = TypedArray<TypedArrayKind::Int8Array>(runtime, size);
                  arr.updateUnsafe(runtime, (int8_t*)mat.data, size * sizeof(int8_t));
                  value.setProperty(runtime, "buffer", arr);
                } else if(type == "int16") {
                  auto arr = TypedArray<TypedArrayKind::Int16Array>(runtime, size);
                  arr.updateUnsafe(runtime, (int16_t*)mat.data, size * sizeof(int16_t));
                  value.setProperty(runtime, "buffer", arr);
                } else if(type == "int32") {
                  auto arr = TypedArray<TypedArrayKind::Int32Array>(runtime, size);
                  arr.updateUnsafe(runtime, (int32_t*)mat.data, size * sizeof(int32_t));
                  value.setProperty(runtime, "buffer", arr);
                } else if(type == "float32") {
                  auto arr = TypedArray<TypedArrayKind::Float32Array>(runtime, size);
                  arr.updateUnsafe(runtime, (float*)mat.data, size * sizeof(float));
                  value.setProperty(runtime, "buffer", arr);
                } else if(type == "float64") {
                  auto arr = TypedArray<TypedArrayKind::Float64Array>(runtime, size);
                  arr.updateUnsafe(runtime, (double*)mat.data, size * sizeof(double));
                  value.setProperty(runtime, "buffer", arr);
                }

                return value;
      });
    } else if (propName == "createObject") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "createObject"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Object {

          return FOCV_Object::create(runtime, arguments, count);
      });
    }
  else if (propName == "toJSValue") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "toJSValue"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Object {

          return FOCV_Object::convertToJSI(runtime, arguments, count);
      });
    }
  else if (propName == "copyObjectFromVector") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "copyObjectFromVector"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Object {

          return FOCV_Object::copyObjectFromVector(runtime, arguments, count);
      });
    }
  else if (propName == "addObjectToVector") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "addObjectToVector"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Value {

          FOCV_Object::addObjectToVector(runtime, arguments, count);

          return jsi::Value(true);
      });
    }
  else if (propName == "invoke") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "invoke"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Object {

          return FOCV_Function::invoke(runtime, arguments, count);
      });
  } else if (propName == "clearBuffers") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "clearBuffers"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Value {
          std::set<std::string> ids_to_keep;

          if (count > 0) {
            auto array = arguments[0].asObject(runtime).asArray(runtime);
            auto length = array.length(runtime);
            for (size_t i = 0; i < length; i++) {
              auto id = array.getValueAtIndex(runtime, i).asString(runtime).utf8(runtime);
              ids_to_keep.insert(id);
            }
          }

          FOCV_Storage::clear(ids_to_keep);
          return true;
      });
  } else if (propName == "siftDetect") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "siftDetect"), 2,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Value {
        try {
          auto origImageId = FOCV_JsiObject::id_from_wrap(runtime, arguments[0]);
          auto origImageRaw = *FOCV_Storage::get<cv::Mat>(origImageId);

          auto maskImageId = FOCV_JsiObject::id_from_wrap(runtime, arguments[1]);
          auto maskImageRaw = *FOCV_Storage::get<cv::Mat>(maskImageId);

          cv::Mat origImageResized = origImageRaw;
          //cv::resize(origImageRaw, origImageResized, cv::Size(1024, 768));

          cv::Mat origImageBW;
          cv::cvtColor(origImageResized, origImageBW, cv::COLOR_BGRA2GRAY);

          cv::Mat origImage = origImageBW;

          cv::Mat maskImageResized = maskImageRaw;
          //cv::resize(maskImageRaw, maskImageResized, cv::Size(1024, 768));

          cv::Mat maskImageBW;
          cv::cvtColor(maskImageResized, maskImageBW, cv::COLOR_BGRA2GRAY);

          cv::Mat maskImage;
          const int lowerBound = 10;
          const int upperBound = 255;
          cv::inRange(maskImageBW, lowerBound, upperBound, maskImage);

          cv::Mat origImageMasked;
          cv::bitwise_or(origImage, origImage, origImageMasked, maskImage);

          cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();

          std::vector<cv::KeyPoint> origKeypoints;
          cv::Mat origDescriptors;

          siftPtr->detectAndCompute(origImage, origImageMasked, origKeypoints, origDescriptors);

          jsi::Object result(runtime);

          auto keypointsId = FOCV_Storage::save(origKeypoints);
          result.setProperty(runtime, "keypoints", FOCV_JsiObject::wrap(runtime, "keypoint_vector", keypointsId));

          auto descriptorsId = FOCV_Storage::save(origDescriptors);
          result.setProperty(runtime, "descriptors", FOCV_JsiObject::wrap(runtime, "mat", descriptorsId));

          return result;
        } catch(const std::exception& ex) {
          __android_log_print(ANDROID_LOG_VERBOSE, "ImageComparison", "The error %s", ex.what());
          throw ex;
        }
    });
  } else if (propName == "siftDrawKeypoints") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "siftDrawKeypoints"), 2,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Value {
        try {
          auto origImageId = FOCV_JsiObject::id_from_wrap(runtime, arguments[0]);
          auto origImageRaw = *FOCV_Storage::get<cv::Mat>(origImageId);

          auto origKeypointsId = FOCV_JsiObject::id_from_wrap(runtime, arguments[1]);
          auto origKeypoints = *FOCV_Storage::get<std::vector<cv::KeyPoint>>(origKeypointsId);

          cv::Mat output;
          drawKeypoints(origImageRaw, origKeypoints, output, cv::Scalar::all(-1));

           cv::Mat result;
           cv::cvtColor(output, result, cv::COLOR_BGRA2RGBA);

           auto id = FOCV_Storage::save(result);
           return FOCV_JsiObject::wrap(runtime, "mat", id);
        } catch(const std::exception& ex) {
          __android_log_print(ANDROID_LOG_VERBOSE, "ImageComparison", "The error %s", ex.what());
          throw ex;
        }
    });
  } else if (propName == "siftCompare") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "siftCompare"), 5,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Value {
        try {
          auto testImageId = FOCV_JsiObject::id_from_wrap(runtime, arguments[0]);
          auto testImageRaw = *FOCV_Storage::get<cv::Mat>(testImageId);

          auto origDescriptorsId = FOCV_JsiObject::id_from_wrap(runtime, arguments[1]);
          auto origDescriptors = *FOCV_Storage::get<cv::Mat>(origDescriptorsId);

          auto origKeypointsId = FOCV_JsiObject::id_from_wrap(runtime, arguments[2]);
          auto origKeypoints = *FOCV_Storage::get<std::vector<cv::KeyPoint>>(origKeypointsId);

          auto origImageId = FOCV_JsiObject::id_from_wrap(runtime, arguments[3]);
          auto origImageRaw = *FOCV_Storage::get<cv::Mat>(origImageId);

          auto maskImageId = FOCV_JsiObject::id_from_wrap(runtime, arguments[4]);
          auto maskImageRaw = *FOCV_Storage::get<cv::Mat>(maskImageId);

          cv::Mat maskImageResized;
          cv::resize(maskImageRaw, maskImageResized, cv::Size(1024, 768));
          cv::Mat maskImageBW;
          cv::cvtColor(maskImageResized, maskImageBW, cv::COLOR_BGRA2GRAY);

          cv::Mat maskImage;
          const int lowerBound = 10;
          const int upperBound = 255;
          cv::inRange(maskImageBW, lowerBound, upperBound, maskImage);

            __android_log_print(ANDROID_LOG_VERBOSE, "TESTTEST", "three");
          cv::Mat testImageBW;
          cv::cvtColor(testImageRaw, testImageBW, cv::COLOR_BGRA2GRAY);

          cv::Mat testImage;
          cv::resize(testImageBW, testImage, cv::Size(1024, 768));

          // cv::Mat testImageMasked;
          // cv::bitwise_or(testImage, testImage, testImageMasked, maskImage);

          cv::Mat origImageResized;
          cv::resize(origImageRaw, origImageResized, cv::Size(1024, 768));

          cv::Mat origImage;
          cv::cvtColor(origImageResized, origImage, cv::COLOR_BGRA2GRAY);

          // cv::Mat origImageMasked;
          // cv::bitwise_or(origImage, origImage, origImageMasked, maskImage);

          cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();

          /*
        std::vector<cv::KeyPoint> fakeOrigKeypoints;
        cv::Mat fakeOrigDescriptors;
        siftPtr->detectAndCompute(origImage, cv::noArray(), fakeOrigKeypoints, fakeOrigDescriptors);

          if (fakeOrigDescriptors.rows == origDescriptors.rows) {
            if (fakeOrigDescriptors.cols == origDescriptors.cols) {
              auto size = fakeOrigDescriptors.total()*fakeOrigDescriptors.elemSize();
              for(auto i=0;i<size;++i) {
                if (fakeOrigDescriptors.data[i] != origDescriptors.data[i]) {
                  __android_log_print(ANDROID_LOG_VERBOSE, "TESTTEST", "DIFFERENT AT: %i", i);
                }
              }
            }
            else {
              __android_log_print(ANDROID_LOG_VERBOSE, "TESTTEST", "DIFFERENT COLS: %i,%i", fakeOrigDescriptors.cols,origDescriptors.cols);
            }
          } else {
            __android_log_print(ANDROID_LOG_VERBOSE, "TESTTEST", "DIFFERENT ROWS: %i,%i", fakeOrigDescriptors.rows,origDescriptors.rows);
          }
          */

        // std::string x;
        //   for(auto i=0;i<fakeOrigDescriptors.rows;++i) {
        //     x += fakeOrigDescriptors.data[i];
        //   }
        // __android_log_print(ANDROID_LOG_VERBOSE, "TESTTEST", "ROW: %d,%d,%d,%d,%d",
        //                     fakeOrigDescriptors.data[0],
        //                     fakeOrigDescriptors.data[1],
        //                     fakeOrigDescriptors.data[2],
        //                     fakeOrigDescriptors.data[3],
        //                     fakeOrigDescriptors.data[4]
        //                     );

        std::vector<cv::KeyPoint> testKeypoints;
        cv::Mat testDescriptors;
        siftPtr->detectAndCompute(testImage, cv::noArray(), testKeypoints, testDescriptors);

        cv::Ptr<cv::flann::KDTreeIndexParams> indexParams(
          new cv::flann::KDTreeIndexParams(5));
        cv::Ptr<cv::flann::SearchParams> searchParams(
          new cv::flann::SearchParams(50));
        cv::Ptr<cv::DescriptorMatcher> matcher(
          new cv::FlannBasedMatcher(indexParams, searchParams));
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(origDescriptors, testDescriptors, knnMatches, 2, cv::noArray());

        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.7f;
        std::vector<cv::DMatch> goodMatches;
        for (size_t i = 0; i < knnMatches.size(); i++) {
          //__android_log_print(ANDROID_LOG_VERBOSE, "TESTTEST", "knn match size: %zu", knnMatches[i].size());
            if (!knnMatches[i].empty()) {
          if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
          }
            }
        }

        cv::Mat output;
        drawMatches(origImage, origKeypoints, testImage, testKeypoints, goodMatches,
        output, cv::Scalar::all(-1), cv::Scalar::all(-1));

          cv::Mat result;

        cv::cvtColor(output, result, cv::COLOR_BGRA2RGBA);
          auto id = FOCV_Storage::save(result);
          return FOCV_JsiObject::wrap(runtime, "mat", id);
        } catch(const std::exception& ex) {
          __android_log_print(ANDROID_LOG_VERBOSE, "TESTTEST", "The error %s", ex.what());
          throw ex;
        }
    });
  }

  return jsi::HostObject::get(runtime, propNameId);
}

std::vector<jsi::PropNameID> OpenCVPlugin::getPropertyNames(jsi::Runtime& runtime) {
    std::vector<jsi::PropNameID> result;

    result.push_back(jsi::PropNameID::forAscii(runtime, "frameBufferToMat"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "base64ToMat"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "matToBuffer"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "createObject"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "toJSValue"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "copyObjectFromVector"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "invoke"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "clearBuffers"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "siftDetect"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "siftDrawKeypoints"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "siftCompare"));

    return result;
}
