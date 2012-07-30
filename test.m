%% Preliminaries

close all
clear all
clc
dbstop if error
% dbstop if warning

% DEFINE RANDOM SEED
rand_seed = 0;

% Set random seed
s = RandStream('mt19937ar', 'seed', rand_seed);
RandStream.setDefaultStream(s);

%% Set up

% Model
dt = 1;
A = [1 dt; 0 1];
C = [1 0];
Q = 0.9*eye(2)+0.1*ones(2);
R = 1;
ds = 2;
do = 1;

% Number of points
N = 5;

% Data
m0 = [0 0]';
P0 = 10*eye(2);
x = zeros(2,N);
y = zeros(1,N);
x0 = mvnrnd(m0, P0)';
last_x = x0;
for kk = 1:N
    x(:,kk) = mvnrnd(last_x, Q);
    y(1, kk) = mvnrnd(C*x(:,kk), R);
    last_x = x(:,kk);
end

%% Normal Kalman filter

tic;

% f_mn = zeros(ds,N);
% f_cov = zeros(ds,ds,N);
% last_mn = m0;
% last_cov = P0;
% for kk = 1:N
%     pred_mn = A*last_mn;
%     pred_cov = A*last_cov*A' + Q;
%     IM = C*pred_mn;
%     IS = (R + C*pred_cov*C');
%     K = pred_cov*C'/IS;
%     f_mn(:,kk) = pred_mn + K * (y(1,kk)-IM);
%     f_cov(:,:,kk) = pred_cov - K*IS*K';
%     last_mn = f_mn(:,kk);
%     last_cov = f_cov(:,:,kk);
% end

[f_mn, f_cov] = kf_loop(m0, P0, C, R, y, A, Q);
[s_mn, s_cov] = rts_smooth(f_mn, f_cov, A, Q);
fprintf('Normal Kalman filter took %fs to do %u times steps.\n', toc, N);

% Plot
figure, hold on
plot(x(1,:), ':k');
plot(y(1,:), 'xr');
plot(s_mn(1,:), 'b');

%% Naive matrix inversion method

tic;

% Define the elements of the inverse covariance matrix
alpha = A'*(Q\A) + inv(Q);
beta = -A'/Q;
gamma = A'*(Q\A) + inv(Q + A*P0*A');
delta = inv(Q);
obs_bit = C'*(R\C);

% Create inverse covariance matrix
invSigma = zeros(N*ds);
invGSGT = zeros(N*ds);
HTinvT = zeros(N*ds, N*do);
F = zeros(N*ds, ds);
for kk = 2:N-1
    invGSGT( ((kk-1)*ds+1):kk*ds, ((kk-1)*ds+1):kk*ds ) = alpha;
    invSigma( ((kk-1)*ds+1):kk*ds, ((kk-1)*ds+1):kk*ds ) = alpha + obs_bit;
end
for kk = 1:N-1
    invGSGT( (kk*ds+1):(kk+1)*ds, ((kk-1)*ds+1):kk*ds ) = beta';
    invGSGT( ((kk-1)*ds+1):kk*ds, (kk*ds+1):(kk+1)*ds ) = beta;
    invSigma( (kk*ds+1):(kk+1)*ds, ((kk-1)*ds+1):kk*ds ) = beta';
    invSigma( ((kk-1)*ds+1):kk*ds, (kk*ds+1):(kk+1)*ds ) = beta;
end
invGSGT( 1:ds, 1:ds ) = gamma;
invGSGT( ((N-1)*ds+1):(N*ds), ((N-1)*ds+1):(N*ds) ) = delta;
invSigma( 1:ds, 1:ds ) = gamma + obs_bit;
invSigma( ((N-1)*ds+1):(N*ds), ((N-1)*ds+1):(N*ds) ) = delta + obs_bit;
for kk = 1:N
    F( ((kk-1)*ds+1):kk*ds, : ) = A^kk;
    HTinvT( ((kk-1)*ds+1):kk*ds, ((kk-1)*do+1):kk*do ) = C'/R;
end

% Solve
cov = inv(invSigma);
mn = invSigma\(invGSGT*F*m0 + HTinvT*reshape(y, N*do, 1));

% Collate the results
nmi_mn = reshape(mn,2,N);
nmi_cov = zeros(2,2,N);
for kk = 1:N
    nmi_cov(:,:,kk) = cov( (ds*(kk-1)+1):ds*kk,(ds*(kk-1)+1):ds*kk );
end

fprintf('Naive batch method took %fs to do %u times steps.\n', toc, N);

% Plot it
plot(nmi_mn(1,:), '-.g');

%% Clever matrix inversion method

tic;

% Define the elements of the inverse covariance matrix
alpha = A'*(Q\A) + inv(Q) + C'*(R\C);
beta = -A'/Q;
gamma = A'*(Q\A) + inv(Q + A*P0*A') + C'*(R\C);
delta = inv(Q) + C'*(R\C);

% mu1_coef(:, :, kk) are the coefficients on mu_1 for mu_kk.
% con_coef(:, kk) are the constant terms for mu_kk
mu1_coef = zeros(ds, ds, N);
con_coef = zeros(ds, N);
xi = zeros(ds, N);

% Sequential forwards sweep
xi(:,1) = (C'/R)*y(:,1) + (Q + A*P0*A')\A*m0;
mu1_coef(:,:,2) = - beta\gamma;
con_coef(:,2) = beta\xi(:,1);
for kk = 2:N-1
    xi(:,kk) = (C'/R)*y(:,kk);
    mu1_coef(:,:,kk+1) = beta\(-alpha*mu1_coef(:,:,kk) - beta'*mu1_coef(:,:,kk-1));
    con_coef(:,kk+1) = beta\(xi(:,kk) - alpha*con_coef(:,kk) - beta'*con_coef(:,kk-1));
end
xi(:,N) = (C'/R)*y(:,N);
final_mu1_coef = delta\(-beta'*mu1_coef(:,:,N-1));
final_con_coef = delta\(xi(:,N)-beta'*con_coef(:,N-1));

% Solve the simultaneous equations
mu1 = (final_mu1_coef-mu1_coef(:,:,N))\(con_coef(:,N)-final_con_coef);

% Back substitution
cmi_mn = zeros(ds,N);
cmi_mn(:,1) = mu1;
for kk = 2:N
    cmi_mn(:,kk) = mu1_coef(:,:,kk)*mu1 + con_coef(:,kk);
end

fprintf('Clever batch method took %fs to do %u times steps.\n', toc, N);

% Plot it
plot(cmi_mn(1:2:end), '-.c');

%% Clever Fourier Transform matrix inversion method

% tic;
% 
% % Create augmented matrixes
% F = zeros(N*ds,ds);
% G = zeros(N*ds);
% H = zeros(N*do, N*ds);
% S = zeros(N*ds,N*ds);
% T = zeros(N*do,N*do);
% GSGTinv = zeros(N*do,N*do);
% for kk = 1:N
%     F( ((kk-1)*ds+1):kk*ds, :) = A^kk;
%     H( ((kk-1)*do+1):kk*do, ((kk-1)*ds+1):kk*ds ) = C;
%     S( ((kk-1)*ds+1):kk*ds, ((kk-1)*ds+1):kk*ds ) = Q;
%     T( ((kk-1)*do+1):kk*do, ((kk-1)*do+1):kk*do ) = R;
%     if kk < N
%         GSGTinv( ((kk-1)*ds+1):kk*ds, ((kk-1)*ds+1):kk*ds ) = A'*(Q\A)+inv(Q);
%         GSGTinv( (kk*ds+1):(kk+1)*ds, ((kk-1)*ds+1):kk*ds ) = -(Q\A);
%         GSGTinv( ((kk-1)*ds+1):kk*ds, (kk*ds+1):(kk+1)*ds ) = -(A'/Q);
%     else
%         GSGTinv( ((kk-1)*ds+1):kk*ds, ((kk-1)*ds+1):kk*ds ) = inv(Q);
%     end
% end
% 
% % Define intermediates
% a = A'*(Q\A)+inv(Q) + C'*(R\C);
% b = -Q\A;
% 
% % Constants
% rt = exp(2*pi*1i/N);
% 
% % We want to invert a banded block-toeplitz matrix (the joint precision
% % matrix). To do this, convert to the equivalent circulant matrix
% 
% % Calculate block-diagonal diagonalised form of equivalent circulant matrix
% % and then invert each block diagonal
% DiagCirc = zeros(ds,ds,N);
% InvDiagCirc = zeros(ds,ds,N);
% for kk = 1:N  % <--- N-way parallelisation!
%     DiagCirc(:,:,kk) = ( a + b'*rt^(kk-1) + b*rt^((kk-1)*(N-1)) )*eye(ds);
%     InvDiagCirc(:,:,kk) = inv(DiagCirc(:,:,kk));
% end
% 
% % Calculate blocks of the inverse of the equivalent circulant matrix, which
% % is itself block-circulant
% InvCirc = zeros(ds,ds,N);
% for kk = 1:N  % <--- This is the bottle neck! :-S
%     for jj = 1:N
%         InvCirc(:,:,kk) = InvCirc(:,:,kk) + (1/N)*(rt^((kk-1)*(N-jj+1)))*InvDiagCirc(:,:,jj);
%     end
% end
% InvCirc = real(InvCirc);
% 
% % Step 2
% VTinvC = zeros(2*ds, N*ds);
% for kk = 1:N
%     VTinvC(1:ds, ((kk-1)*ds+1):kk*ds) = b*InvCirc(:,:, mod(kk,N)+1 );
%     VTinvC(ds+1:2*ds, ((kk-1)*ds+1):kk*ds) = InvCirc(:,:,kk) - Q*b*InvCirc(:,:, mod(kk,N)+1 );
% end
% 
% % Step 3
% inner = eye(2*ds) - [b*InvCirc(:,:,2), b*InvCirc(:,:,1)*b';
%                      InvCirc(:,:,1)-Q*b*InvCirc(:,:,2), InvCirc(:,:,N)*b'-Q*b*InvCirc(:,:,1)*b'];
% inv_inner = inv(inner);
% 
% % Step 5
% invCU = zeros(N*ds,2*ds);
% for kk = 1:N
%     invCU( ((kk-1)*ds+1):kk*ds, 1:ds ) = InvCirc(:,:, mod(kk,N-kk+1)+1 );
%     invCU( ((kk-1)*ds+1):kk*ds, ds+1:2*ds) = InvCirc(:,:, N-kk+1 );
% end
% 
% % Calculate the marginal covariances
% cmi_cov = zeros(2,2,N);
% for kk = 1:N
%     cmi_cov(:,:,kk) = InvCirc(:,:,1) + (...
%         invCU( ((kk-1)*ds+1):kk*ds, 1:ds ) * inv_inner(1:ds,1:ds) * VTinvC( 1:ds, ((kk-1)*ds+1):kk*ds ) + ...
%         invCU( ((kk-1)*ds+1):kk*ds, 1:ds ) * inv_inner(1:ds,ds+1:2*ds) * VTinvC( ds+1:2*ds, ((kk-1)*ds+1):kk*ds ) + ...
%         invCU( ((kk-1)*ds+1):kk*ds, ds+1:2*ds ) * inv_inner(ds+1:2*ds,1:ds) * VTinvC( 1:ds, ((kk-1)*ds+1):kk*ds ) + ...
%         invCU( ((kk-1)*ds+1):kk*ds, ds+1:2*ds ) * inv_inner(ds+1:2*ds,ds+1:2*ds) * VTinvC( ds+1:2*ds, ((kk-1)*ds+1):kk*ds ) );
% end
% 
% % Construct the inverse (because we're not going for speed yet)
% InvCircFull = zeros(ds*N,ds*N);
% for ii = 1:N
%     for jj = 1:N
%         kk = mod(ii-jj,N)+1;
%         InvCircFull((ds*(ii-1)+1):ds*ii,(ds*(jj-1)+1):ds*jj) = InvCirc(:,:,kk);
%     end
% end
% 
% % Now calculate the covariance matrix
% Sigma = real(InvCircFull + invCU*(inner\VTinvC));
% 
% % % Now calculate the covariance matrix
% % U = [eye(ds) zeros(ds); zeros((N-2)*ds,2*ds); zeros(ds) b'];
% % V = [zeros(ds) eye(ds); zeros((N-2)*ds,2*ds); b' b'*Q];
% % Sigma = real(InvCircFull + InvCircFull*U/(eye(2*ds)-V'*InvCircFull*U)*V'*InvCircFull);
% 
% % And now the mean
% mu = Sigma*(GSGTinv*F*x0 + H'*(T\y'));
% 
% % Collate the results
% cmi_mn = reshape(mu,2,N);
% % cmi_cov = zeros(2,2,N);
% % for kk = 1:N
% %     cmi_cov(:,:,kk) = Sigma( (ds*(kk-1)+1):ds*kk,(ds*(kk-1)+1):ds*kk );
% % end
% 
% 
% fprintf('Complicated batch method took %fs to do %u times steps.\n', toc, N);
% 
% % Plot it
% plot(mu(1:2:end), '-.c');

%% Compare

figure, hold on
plot( log(abs(nmi_mn(1,:)-s_mn(1,:))) , 'b');
plot( log(abs(cmi_mn(1,:)-s_mn(1,:))) , 'r');

% kf_det = zeros(1,N);
% nmi_det = zeros(1,N);
% cmi_det = zeros(1,N);
% for kk = 1:N
%     kf_det(kk) = det(s_cov(:,:,kk));
%     nmi_det(kk) = det(nmi_cov(:,:,kk));
%     cmi_det(kk) = det(cmi_cov(:,:,kk));
% end
% 
% figure, hold on
% plot(kf_det, 'k')
% plot(nmi_det, 'b')
% plot(cmi_det, 'r')
% 
% figure, hold on
% plot( log(abs(nmi_det-kf_det)), 'b')
% plot( log(abs(cmi_det-kf_det)), 'r')