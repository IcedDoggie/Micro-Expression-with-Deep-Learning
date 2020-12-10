function [opticalStrain, strain_orientation, exx, eyy] = os(optical_flow) 

%% tmp is optical flow map.
u = optical_flow(:,:,1); % horizontal OF
v = optical_flow(:,:,2); % vertical OF

[x,y]=size(u);
u_x = u(:,1:y) - u(:,[1,1:y-1]); 
v_y = v(1:x,:) - v([1,1:x-1],:);
u_y = u(1:x,:) - u([1,1:x-1],:);
v_x = v(:,1:y) - v(:,[1,1:y-1]);

% Compute Strain Orientation
exx = u_x
eyy = v_y
exy = 0.5 * (u_y + v_x)
de_zero = 0.0000001

theta_p = atan( 2*exy./(exx-eyy+de_zero) ) / 2

e_xx_cos = exx .* cos(theta_p).^2
e_xy_sin = exy .* sin(theta_p).^2
e_yy_cos = eyy .* cos(theta_p).^2



strain_orientation = sqrt( (e_xx_cos.^2) + (e_yy_cos.^2) + 2 * (e_xy_sin.^2)  )
u_x=u(:,1:y)-u(:,[1,1:y-1]);
v_y=v(1:x,:)-v([1,1:x-1],:);
u_y=u(1:x,:)-u([1,1:x-1],:);
v_x=v(:,1:y)-v(:,[1,1:y-1]);

opticalStrain = sqrt((u_x.^2)+v_y.^2+1/2*(u_y+v_x).^2);

%opticalStrain = sqrt( (exx.^2) + eyy.^2 + (exy).^2 + (exy).^2 );  
OFmagnitude = sqrt(u.^2 + v.^2);
OForientation = atan2(v,u)*180/pi;

endfunction