classdef ImuFusion < handle
    
    properties
        Accel
        Gyro
        Camera
        FUSE
        update_count = 0;
        track_orientation = quaternion(1,0,0,0);
    end
    
    methods
        function obj = ImuFusion()
            obj.Accel = insAccelerometer;
            obj.Gyro = insGyroscope;
            obj.Camera = insCamera;
            filt = insEKF(obj.Accel, obj.Gyro, obj.Camera, insMotionPose);
            si = stateinfo(filt);
            processNoise = diag(filt.AdditiveProcessNoise)*0;
            processNoise(si.Position) = 0;
            processNoise(si.Velocity) = 1e-4;
            processNoise(si.Acceleration) = 10;
            processNoise(si.AngularVelocity) = 10;
            processNoise(si.Orientation) = 1e-4;
            processNoise(si.Accelerometer_Bias) = 5e-4;
            processNoise(si.Gyroscope_Bias) = 0;
            filt.AdditiveProcessNoise = diag(processNoise);
            obj.FUSE = filt;
            reset_state(obj);
        end
        
        function [position, orientation] = step(obj,accel,gyro,dt)
            arguments (Input)
                obj (1,1) ImuFusion
                accel (:,3) {mustBeReal}
                gyro (:,3) {mustBeReal}
                dt (1,1)
            end

            accel = [-accel(:,2), -accel(:,1), accel(:,3)];
            gyro = [gyro(:,2), gyro(:,1), -gyro(:,3)];

            predict(obj.FUSE,dt);
            fuse(obj.FUSE, obj.Gyro, gyro, 1e-6);
            fuse(obj.FUSE, obj.Accel, accel, 1e-3);

            position = stateparts(obj.FUSE,"Position");
            orientation = stateparts(obj.FUSE,"Orientation");
            orientation_quat = quaternion(orientation);

            tip_pos = position + rotatepoint(orientation_quat, [0 .143 0]);
            position = tip_pos;

            obj.update_count = obj.update_count + 1;

            check_divergence(obj);
        end

        function update_tracker(obj, orientation_mat, tip_pos_opencv)
            or_quat = get_orientation_quat(obj,orientation_mat);
            obj.track_orientation = or_quat;
            tip_pos = [tip_pos_opencv(1), -tip_pos_opencv(2), -tip_pos_opencv(3)];
            imu_pos = tip_pos - rotatepoint(or_quat, [0 .143 0]);

            cov_pos = 0.3e-6;
            cov_or = 0.5e-5;
            fuse(obj.FUSE, obj.Camera, [imu_pos compact(or_quat)].', [cov_pos cov_pos cov_pos cov_or cov_or cov_or cov_or]);
        end

        function quat = get_orientation_quat(~,orientation_mat_opencv)
            orientation_mat = orientation_mat_opencv(:,[3 2 1]) .* [-1; 1; 1];
            quat = quaternion(orientation_mat, "rotmat", "point");
        end

        function reset_state(obj)
            filt = obj.FUSE;
            filt.State = filt.State*0;
            stateparts(filt, "Orientation", [1 0 0 0]);
            obj.FUSE.StateCovariance = filt.StateCovariance .* 0;
            statecovparts(filt, "Orientation", 0.3);
            obj.FUSE = filt;
        end

        function check_divergence(obj)
            if max(obj.FUSE.State, [], 'all') > 1e2 || max(obj.FUSE.StateCovariance, [], 'all') > 1e2
                disp("Resetting filter");
                reset_state(obj);
            end
        end
    end
end

