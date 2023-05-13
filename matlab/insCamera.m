classdef insCamera < positioning.INSSensorModel

    methods 
        function z = measurement(~, filt)
            %MEASUREMENT Sensor measurement estimate from states
            %  MEASUREMENT returns an M-by-1 array of predicted measurements
            %  for this SENSOR based on the current state of filter FILT. Here
            %  M is the number of elements in a measurement from this sensor.
            %  The FILT input is an instance of the insEKF filter.
            %
            %   This function is called internally by FILT when its FUSE or
            %   RESIDUAL methods are invoked.

            idx = stateinfo(filt);
            state = filt.State;
            pos = state(idx.Position);
            % orientation = [0 0 0 0];
            orientation = state(idx.Orientation);
            z = [pos(:); orientation(:)].';
        end

        function dhdx = measurementJacobian(~, filt)
        %MEASUREMENTJACOBIAN Jacobian of the measurement method
        %  MEASUREMENTJACOBIAN returns a 3-by-NS array if the filter FILT's
        %  motion model tracks position but not velocity.
        %
        %  MEASUREMENTJACOBIAN returns a 6-by-NS array if the filter FILT's
        %  motion model tracks both position and velocity.
        %
        %  The returned matrix is the Jacobian of the MEASUREMENT method
        %  relative to the State property of filter FILT. The FILT input is
        %  an instance of the insEKF filter. Here NS is the number of
        %  elements in the State property of FILT. 
        %
        %   This function is called internally by FILT when its FUSE or
        %   RESIDUAL methods are invoked.
        %
        %   See also: insEKF, insMotionPose
            idx = stateinfo(filt);
            state = filt.State;
            pidx = idx.Position;
            oidx = idx.Orientation;

            dhdx = zeros(7, numel(state), 'like', state);
            dhdx(1,pidx) = [1 0 0];
            dhdx(2,pidx) = [0 1 0];
            dhdx(3,pidx) = [0 0 1];

            dhdx(4,oidx) = [1 0 0 0];
            dhdx(5,oidx) = [0 1 0 0];
            dhdx(6,oidx) = [0 0 1 0];
            dhdx(7,oidx) = [0 0 0 1];
        end
    end

end
