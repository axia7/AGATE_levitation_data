function run_AGATE_weights(att_ind, n_its, pid, nc_bool, W_poinc, W_shape)

% run AGATE method with weights on loss function terms

%{
att_ind (int): index of data attractor to run on (1 to 55)
n_its (int): number of fminsearch iterations to run
pid (int): random seed to generate noise in initial coefficients, 1 corresponds to no noise
nc_bool (bool): if 0, full equation, if 1, no nonlinear couplings
W_poinc (float): weight to multiply L_poinc
W_shape (float): weight to multiple W_shape
%}

if nc_bool
    fprintf("currently running NO COUPLE on attractor %.0f \n", att_ind)
else
    fprintf("currently running FULL EQN on attractor %.0f \n", att_ind)
end

% input
attractors = reshape(readmatrix('attractors.txt'), 55, 4, 15000);
sample_att = squeeze(attractors(att_ind, :, :));
y0_shifts = readmatrix('y0shift.txt');
g_val = y0_shifts(att_ind);

clear a, clear coeffs

if nc_bool
    a = [sym('y') sym('t') sym('dy') sym('dt')];
    f = [1 sym('t') sym('y') sym('dt')];  % ddtheta fitting library
    f_y = [1 sym('y') sym('y').^3 sym('dy')];  % ddy fitting library
    mF = matlabFunction(f(2:end), 'Vars', a);
    mF_y = matlabFunction(f_y(2:end), 'Vars', a);   
else
    a = [sym('y') sym('t') sym('dy') sym('dt')];
    f = [1 sym('t') sym('y') sym('t').*sym('y').^2 sym('dt') sym('dt').^2.*sym('y') sym('dt').*sym('dy').*sym('y')];  % ddtheta fitting library
    f_y = [1 sym('y') sym('y').^3 sym('dy')];  % ddy fitting library
    mF = matlabFunction(f(2:end), 'Vars', a); 
    mF_y = matlabFunction(f_y(2:end), 'Vars', a); 
end

% filter out nan padding
sample_att = sample_att(:, ~isnan(sample_att(1, :)));
att_len = length(sample_att);
fprintf("video length is %.0f \n", att_len)

% generate coefficients via least squares regression
yf = sample_att(1, :)'; dyf = sample_att(3, :)';
tf = sample_att(2, :)'; dtf = sample_att(4, :)';

ddyf = gradient(dyf);
ddtf = gradient(dtf);

M_y = [ones(size(yf,1),1) mF_y(yf, tf, dyf, dtf)]; % all 3rd order terms (timeseries)
M = [ones(size(yf,1),1) mF(yf, tf, dyf, dtf)]; % all 3rd order terms (timeseries)

coeffs_t = lsqminnorm(M, ddtf)'; % coefficient solutions to d2y and d2theta systems
coeffs_y = lsqminnorm(M_y, ddyf)';
init_coeffs = [coeffs_y coeffs_t];

% regenerate mF mapper to include constant accelerations
mF_y = matlabFunction(f_y(1:end), 'Vars', a); 
mF = matlabFunction(f(1:end), 'Vars', a);

% generate data poincare sections
a = [1 0 0 0]; b = 0;
test_traj = [sample_att(1, :); sample_att(2, :); sample_att(3, :); sample_att(4, :)];
test_traj_y0 = [sample_att(1, :)-nanmean(sample_att(1, :)); sample_att(2, :); sample_att(3, :); sample_att(4, :)];

proj = ((a*test_traj_y0)-b)./(norm(a).^2); % off-plane projection, y=0 plane
sign_swaps = find((diff(sign(proj)) == 2)); % pick y=0 crossing where dy > 0
data_section = (test_traj_y0(:, sign_swaps) + test_traj_y0(:, sign_swaps+1))./2;

data_att.ydata = [sample_att'];
data_att.ylabels = {'y', 'th', 'dy', 'dth'};

% use fmin search to optimize coefficients

c00 = init_coeffs;
if att_len > 12000
    n_evo = 10000; % number of time points to evolve for
    n_sec = 20; % number of poincare sections to evaaluate loss over
else
    n_evo = round(att_len*0.85); % set evolution to 85% of trajectory length
    n_sec = min([round(n_evo / 150), 20]); 
end

final_results = zeros(n_its, length(f)+length(f_y)+3);

% while loop to generate coefficients with noise that don't blow up and have poincare section points

x_sim_fmin = 1e4; % initialize variable for integrated trajectories
n_comp = 0; % while loop to generate random coefficients
n_gen = 0;

eta = 0.05; % random coefficient noise

rng(pid);
fprintf(" attractor k=%.0f ** currently on ic %.0f ** \n \n", att_ind, pid)
if pid == 1
    c00 = init_coeffs;
else
    while n_comp < 30 || max(x_sim_fmin, [], 'all') > 1e2 || nanbool == 1
        fprintf("generating coefficients \n")

        c00 = init_coeffs .* (1+rand(size(init_coeffs))*2*eta - eta); % up to 1 percent flucts
        c00(1) = g_val*c00(2);
        % if nc_bool 
        %     c00([8 10 11]) = 0;
        % end

        coeffs_y = c00(1:length(f_y)); % coeffs y      
        coeffs_th = c00(length(f_y)+1:end); % coeffs theta   
        
        coeffs_th(1) = g_val*coeffs_th(3);
      
        % rk ode integrator
        [x_sim_fmin,~, ~, ~] = rk_full_edit(data_att.ydata', 1, coeffs_th', coeffs_y', mF, mF_y, n_evo);            

        x_sim_fmin_y0 = [x_sim_fmin(1, :)-nanmean(x_sim_fmin(1, :)); x_sim_fmin(2, :); x_sim_fmin(3, :); x_sim_fmin(4, :)];        
        
        a = [1 0 0 0]; b = 0;
        proj = ((a*x_sim_fmin_y0)-b)./(norm(a).^2); % off-plane projection, y=0 plane
        sign_swaps = find((diff(sign(proj)) == 2)); % pick y=0 crossing where dy > 0
        n_comp = length(sign_swaps); % number of poincare section points
        nanbool = any(isnan(x_sim_fmin)==1, 'all'); % any nans
        n_gen = n_gen + 1;
        if n_gen >= 500
            fprintf("spent too long generating, quitting \n")
            return
        end
    end
end

fprintf("max point is %.2e \n", max(abs(x_sim_fmin),[],'all'))

fprintf("done generating coefficients n_poinc= %.0f \n", n_comp)

L = loss_function(c00, data_att, 1, n_sec, n_evo, nc_bool, g_val, W_shape, W_poinc);

if isnan(L)
    fprintf("loss function is nan, quitting \n")
    return
end

fprintf("initial loss function is %.2e \n", L)
fprintf("c00 length is %.0f \n", length(c00))

for kk = [1:n_its]

    options_f = optimset('MaxIter', 100);
    tic
    if kk == 1
        % built in fminsearch optimizer
        [c0,ss0] = fminsearch(@loss_function, c00, options_f, data_att, 0, n_sec, n_evo, nc_bool, g_val, W_shape, W_poinc);
    else
        [c0,ss0] = fminsearch(@loss_function, c0, options_f, data_att, 0, n_sec, n_evo, nc_bool, g_val, W_shape, W_poinc);
    end    
    toc

    fprintf("kk is %.0f, c0 len is %.0f \n", kk, length(c0))

    c0(1) = g_val*c0(2);    

    % if nc_bool 
    %     c0([6 8 9]) = 0;
    % end

    L = loss_function(c0, data_att, 1, n_sec, n_evo, nc_bool, g_val, W_shape, W_poinc);

    if isnan(L)
        fprintf("loss function is nan, quitting \n")
        return
    end    

    coeffs_y = c0(1:length(f_y)); % coeffs y    
    coeffs_th = c0(length(f_y)+1:end); % coeffs theta
    coeffs_th(1) = g_val*coeffs_th(3);
    
    % time evolve
    [x_sim_fmin,~, ~, ~] = rk_full_edit(data_att.ydata', 1, coeffs_th', coeffs_y', mF, mF_y, n_evo);   

    n_poinc = n_sec; % number of poincare error points (to evaluate L)
    n_haus = 1000; % number of hausdorff error points (to evaluate L)

    % find sim poincare section
    a = [1 0 0 0]; b = 0;
    max_cyc = 100;
    test_traj = [x_sim_fmin(1, :); x_sim_fmin(2, :); x_sim_fmin(3, :); x_sim_fmin(4, :)];
    test_traj_y0 = [x_sim_fmin(1, :)-nanmean(x_sim_fmin(1, :)); x_sim_fmin(2, :); x_sim_fmin(3, :); x_sim_fmin(4, :)];        

    proj = ((a*test_traj_y0)-b)./(norm(a).^2); 
    sign_swaps = find((diff(sign(proj)) == 2)); 
    p_section = (test_traj_y0(:, sign_swaps) + test_traj_y0(:, sign_swaps+1))./2; % poincare section: avg of nearest {y<0, y>0} crossing

    sim_section_cut = p_section([2 4], [1:n_poinc]); % {theta, dtheta plane at y=0}

    % find data poincare section
    x_data = sample_att(:, 1:att_len);
    x_data_y0 = x_data; % data in the frame mean(y) = 0
    x_data_y0(1, :) = x_data_y0(1, :) - nanmean(x_data_y0(1, :));

    proj_data = ((a*x_data_y0)-b)./(norm(a).^2); 
    sign_swaps_data = find((diff(sign(proj_data)) == 2)); 
    p_section_data = x_data_y0(:, sign_swaps_data);
    data_section_cut = p_section_data([2 4], [1:n_poinc]);

    % find poincare error
    sec_diff = sim_section_cut - data_section_cut;
    poinc_error = vecnorm(sec_diff ./ max(abs(data_section_cut), [], 2), 2, 1).^2;

    % find hausdorff error
    hausdorff = zeros(1, n_haus);
    attractor_y0 = [sample_att(1, :)-nanmean(sample_att(1, :)); sample_att(2, :); sample_att(3, :); sample_att(4, :)]';
    count = 1;

    if length(attractor_y0) > 8e3
        for jj = [5e3:5e3+n_haus]
            current_pt = test_traj_y0(:, jj)';
            att_diffs = vecnorm((attractor_y0(5e3:8e3, :) - current_pt)./max(abs(attractor_y0)), 2, 2).^2;
            min_d = min(att_diffs);
            hausdorff(count) = min_d;
            count = count + 1;
        end
    else
        att_length = length(attractor_y0);
        for jj = [att_length-2e3-1:att_length-1e3-1]
            current_pt = test_traj_y0(:, jj)';
            att_diffs = vecnorm((attractor_y0(att_length-2e3-1:att_length-1, :) - current_pt)./max(abs(attractor_y0)), 2, 2).^2;
            min_d = min(att_diffs);
            hausdorff(count) = min_d;
            count = count + 1;
        end
    end        

    h_err = nanmean(hausdorff(1:n_haus));
    p_err = nanmean(poinc_error(1:n_poinc));

    fprintf("L is %.2e || h err is %.2e || p err is %.2e \n \n", L, h_err, p_err)
    final_results(kk, :) = [c0, L, h_err, p_err];
end

fprintf("done running analysis \n")

dir_str = strcat('optimized_dh', num2str(W_shape), '_dp_', num2str(W_poinc), '/');
if ~exist(dir_str, 'dir')
    mkdir(dir_str)
end 

if nc_bool
    writematrix(final_results, strcat(dir_str, 'fitted_NC_', num2str(att_ind),'_pid_', num2str(pid), '.txt'))
else
    writematrix(final_results, strcat(dir_str, 'fitted_full_', num2str(att_ind), '_pid_', num2str(pid), '.txt'))
end

fprintf("success writing to file!!")

end

function [L] = loss_function(coeffs_i, data, print_bool, n_sections, num_evolve, nc_bool, g_val, W_shape, W_poinc)

if nc_bool
    clear a, clear coeffs
    a = [sym('y') sym('t') sym('dy') sym('dt')];
    f = [1 sym('t') sym('y') sym('dt')];  % ddtheta fitting library
    f_y = [1 sym('y') sym('y').^3 sym('dy')];  % ddy fitting library
    mF = matlabFunction(f(1:end), 'Vars', a); % omit constant theta term
    mF_y = matlabFunction(f_y(1:end), 'Vars', a); % include constant y term (offset y0 ~ -g/k)    
else
    clear a, clear coeffs
    a = [sym('y') sym('t') sym('dy') sym('dt')];
    f = [1 sym('t') sym('y') sym('t').*sym('y').^2 sym('dt') sym('dt').^2.*sym('y') sym('dt').*sym('dy').*sym('y')];  % ddtheta fitting library
    f_y = [1 sym('y') sym('y').^3 sym('dy')];  % ddy fitting library
    mF = matlabFunction(f(1:end), 'Vars', a); % omit constant theta term
    mF_y = matlabFunction(f_y(1:end), 'Vars', a); % include constant y term (offset y0 ~ -g/k)
end

coeffs_i(1) = g_val*coeffs_i(2);

x_data = data.ydata';
x_data_y0 = x_data;
x_data_y0(1, :) = x_data(1, :) - nanmean(x_data(1, :)); 

coeffs_th = coeffs_i(length(f_y)+1:end); 
coeffs_th(1) = g_val*coeffs_th(3);
coeffs_y = coeffs_i(1:length(f_y));

% integrate
[x_sim,~, ~, ~] = rk_full_edit(x_data, 1, coeffs_th', coeffs_y', mF, mF_y, num_evolve);   

% y0 frame
x_sim_y0 = x_sim;
x_sim_y0(1, :)  = x_sim_y0(1, :) - nanmean(x_sim_y0(1, :));

% period term L_T: keep commesurate y periods
y_sim = x_sim(1, :); [~, y_locs_sim] = findpeaks(y_sim); 
y_data = x_data(1, :); [~, y_locs_data] = findpeaks(y_data);
T_y_sim = mean(diff(y_locs_sim)); T_y_data = mean(diff(y_locs_data));

L_T = (T_y_sim - T_y_data).^2 ./ (T_y_data^2);

% poincare term: find poincare section
a = [1 0 0 0]; b = 0;
proj = ((a*x_sim_y0)-b)./(norm(a).^2); % off-plane projection
sign_swaps = find((diff(sign(proj)) == 2)); % pick one ordered crossing, NOTE: try the other
p_section_sim = x_sim_y0(:, sign_swaps);

a = [1 0 0 0]; b = 0;
proj_data = ((a*x_data_y0)-b)./(norm(a).^2); % off-plane projection
sign_swaps_data = find((diff(sign(proj_data)) == 2)); % pick one ordered crossing, NOTE: try the other
p_section_data = x_data_y0(:, sign_swaps_data);

% normalize th/dth plane by data max values
max_scale = max(abs(p_section_data), [], 2);

if length(p_section_sim) < n_sections % return large error if simulated trace doesn't converge
    L = 1e3 * (n_sections - size(p_section_sim, 2));
    return
end

norm_p_sim = p_section_sim([2 4], 1:n_sections)./ max_scale([2 4]);
norm_p_data = p_section_data([2 4], 1:n_sections)./ max_scale([2 4]);

% evaluate hausdorff error
hausdorff = zeros(1, 500);
count = 1;

if length(x_data_y0) >= 6500
    scale = max(abs(x_data_y0), [], 2);
    for jj = round(linspace(5e3, 5e3+1000, 500))
        current_pt = x_sim_y0(:, jj);
        att_diffs = vecnorm((x_data_y0(:, 5e3:8e3) - current_pt)./ scale, 2, 1).^2;
        min_d = min(att_diffs);
        hausdorff(count) = min_d;
        count = count + 1;
    end
else
    new_start = length(x_data_y0) - 1500;
    scale = max(abs(x_data_y0), [], 2);
    for jj = round(linspace(new_start, new_start+1000, 500))
        current_pt = x_sim_y0(:, jj);
        att_diffs = vecnorm((x_data_y0(:, new_start-1499:new_start-1) - current_pt)./ scale, 2, 1).^2;
        min_d = min(att_diffs);
        hausdorff(count) = min_d;
        count = count + 1;
    end    
end

L_poinc = W_poinc .* mean(vecnorm(norm_p_sim - norm_p_data, 2, 1).^2);
L_hd = W_shape .* 10 * nanmean(hausdorff);
L_T = L_T * 1e3;
L = L_hd + L_T + L_poinc;

if print_bool
    fprintf("L IS %.2e || hd %.2e || T %.2e || P %.2e \n", L, L_hd, L_T, L_poinc)
end

end