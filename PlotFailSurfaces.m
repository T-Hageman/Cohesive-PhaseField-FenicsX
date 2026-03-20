clear all
close all
clc

% Publication-quality defaults
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesFontSize', 9);
set(groot, 'defaultLineLineWidth', 1.2);

% Colorblind-friendly palette (adapted from Tol's muted scheme)
clrs = distinguishable_colors(6);

modes = {"r1","DP","DP","DP","DP","DP"};
legendLabels = {"r1";
	"DP, $\varepsilon_\mathrm{ref}=10^{-4}$";
	"DP, $\varepsilon_\mathrm{ref}=5\cdot 10^{-4}$";
	"DP, $\varepsilon_\mathrm{ref}=10^{-3}$";
	"DP, $\varepsilon_\mathrm{ref}=5\cdot 10^{-3}$";
	"DP, $\varepsilon_\mathrm{ref}=10^{-2}$"};
dpRef = [1, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2];
fig = figure(1);
for zzz=1:length(modes)
	mode=modes{zzz};
	lName = legendLabels{zzz};
	dp_eRef = dpRef(zzz);

	ft = 150e6;

	smin = -3e8;
	smax = 1.5e8;
	nPoints = 51;
	
	L = 115384615384.61537;
	poisson = 0.3;
	G = L*(1/(2*poisson)-1);
	K = L + 2*G/3;
	D = [L+2*G L L; L L+2*G L; L L L+2*G];
	Dinv = inv(D);
	
	sRange = linspace(smin, smax, nPoints);
	[s1,s2,s3] = meshgrid(sRange, sRange, sRange);
	F = zeros(nPoints, nPoints, nPoints);
	F_PlaneStrain = zeros(nPoints, nPoints);
	F1m = zeros(nPoints, nPoints);
	F2m = zeros(nPoints, nPoints);
	F_J2 = zeros(nPoints, nPoints);
	F1IJm = zeros(nPoints, nPoints);
	F2IJm = zeros(nPoints, nPoints);
	
	if strcmp(mode, "r1")
		modeId = 1;
		dp_eRef = 0.0;
	elseif strcmp(mode, "r2")
		modeId = 2;
		dp_eRef = 0.0;
	else
		modeId = 3;
	end

	stress = zeros(3,1);
	strain = zeros(3,1);
	for i=1:nPoints
		for j=1:nPoints
			for k=1:nPoints
				stress(1) = s1(i,j,k);
				stress(2) = s2(i,j,k);
				stress(3) = s3(i,j,k);
				strain = Dinv*stress;
	
				[F1,F2]  = Get_F(strain, stress, modeId, ft, dp_eRef);
				F(i,j,k) = min(F1, F2);
			end
			stress(1) = s1(i,j,1);
			stress(2) = s2(i,j,1);
			stress(3) = poisson*(stress(1)+stress(2));
			strain = Dinv*stress;
			[F1,F2]  = Get_F(strain, stress, modeId, ft, dp_eRef);
			F_PlaneStrain(i,j) = min(F1, F2);
			F1m(i,j) = F1;
			F2m(i,j) = F2;
		end
	end
	
	subplot(1,3,1)
		srf = isosurface(s1/ft,s2/ft,s3/ft,F,0.0);
		p = patch(srf,'FaceColor',clrs(zzz,:));
		p.EdgeColor = 'none';
		p.FaceAlpha = 0.25;
		hold on

		xlim([smin, smax]/ft)
		ylim([smin, smax]/ft)
		zlim([smin, smax]/ft)
		grid on
		axis equal

		ax1 = gca;
		ax1.LineWidth = 0.8;
		ax1.GridAlpha = 0.15;
		ax1.GridLineStyle = '-';

		camlight;
		lighting gouraud;
		hold on
		view(45,45)

		% Manually position axis labels to avoid MATLAB's poor 3D placement
		ax1.XLabel.String = '$\sigma_1/f_\mathrm{t}$';
		ax1.YLabel.String = '$\sigma_2/f_\mathrm{t}$';
		ax1.ZLabel.String = '$\sigma_3/f_\mathrm{t}$';
		ax1.XLabel.Rotation = -45;
		ax1.YLabel.Rotation = 45;
		ax1.XLabel.Position(2) = ax1.YLim(1) - 0.35*diff(ax1.YLim);
		ax1.XLabel.Position(3) = ax1.ZLim(1) - 0.1*ax1.ZLim(1);
		ax1.YLabel.Position(1) = ax1.XLim(2) + 0.35*diff(ax1.XLim);
		ax1.YLabel.Position(3) = ax1.ZLim(1) - 0.1*ax1.ZLim(1);
		ax1.XLabel.HorizontalAlignment = 'center';
		ax1.YLabel.HorizontalAlignment = 'center';
		%title('(a) 3D')
	
	subplot(1,3,2)
		s3_zeroIndex = find(s3(1,1,:)>=-1e-6, 1);
		s1PS = s1(:,:,s3_zeroIndex);
		s2PS = s2(:,:,s3_zeroIndex);
		F_PS = F(:,:,s3_zeroIndex);

		contour(s1PS/ft, s2PS/ft, F_PS, [0 0], 'Color', clrs(zzz,:), 'LineWidth', 1.2, 'DisplayName', lName)
		hold on
		grid on
		axis equal
		ax2 = gca;
		ax2.LineWidth = 0.8;
		ax2.GridAlpha = 0.15;
		xlabel('$\sigma_1/f_\mathrm{t}$')
		ylabel('$\sigma_2/f_\mathrm{t}$')
		%title('(b) Plane stress, $\sigma_3=0$')
	
	subplot(1,3,3)
		contour(s1PS/ft, s2PS/ft, F_PlaneStrain, [0 0], 'Color', clrs(zzz,:), 'LineWidth', 1.2, 'DisplayName', lName)
		hold on
		grid on
		axis equal
		ax3 = gca;
		ax3.LineWidth = 0.8;
		ax3.GridAlpha = 0.15;
		xlabel('$\sigma_1/f_\mathrm{t}$')
		ylabel('$\sigma_2/f_\mathrm{t}$')
		%title('(c) Plane strain, $\varepsilon_3=0$')
end

% Create shared legend at the bottom spanning all subplots
subplot(1,3,3);
h = findobj(gca, 'Type', 'Contour');
lg = legend(flip(h), 'Orientation', 'horizontal', 'NumColumns', 3, ...
	'FontSize', 8, 'Box', 'on', 'EdgeColor', [0.5 0.5 0.5]);
lg.Units = 'normalized';
lg.Position = [0.25, 0.1, 0.5, 0.05];

% Shift subplots up to make room for the legend row
for k = 1:3
	sp = subplot(1,3,k);
	pos = sp.Position;
	sp.Position = [pos(1), pos(2) + 0.06, pos(3), pos(4) - 0.04];
end

%saveFigNow(fig, "FailureCrit_Combined", 8, 18)

function saveFigNow(fg, sname, HFig, WFig)
	figure(fg);
	fprintf(sname + "  ")

	% Set figure size
	fg.Units = 'centimeters';
	fg.Position = [2 2 WFig HFig];
	set(fg, 'color', 'w');

	% Ensure consistent font sizing across all subplots
	allAxes = findall(fg, 'Type', 'axes');
	for iax = 1:length(allAxes)
		allAxes(iax).FontSize = 9;
		allAxes(iax).LabelFontSizeMultiplier = 1.0;
		allAxes(iax).TitleFontSizeMultiplier = 1.1;
	end

	% Tight layout via PaperPosition
	fg.PaperUnits = 'centimeters';
	fg.PaperPosition = [0 0 WFig HFig];
	fg.PaperSize = [WFig HFig];

	drawnow();
	print(fg, sname+".png",'-dpng','-r1200'); fprintf(".png  ")
	print(fg, sname+".jpg",'-djpeg','-r1200'); fprintf(".jpg  ")
	print(fg, sname+".eps",'-depsc','-r1200'); fprintf(".eps  ")
	print(fg, sname+".svg",'-dsvg','-r1200'); fprintf(".svg  ")
	print(fg, sname+".emf",'-dmeta','-r1200'); fprintf(".emf\n")
end

function [Failure1, Failure2] = Get_F(strain, stress, modeId, ft, dp_eRef)
	fs = 0.5*ft;

	i = [1 1 1];
	i_col = i';
	J = [2/3 -1/3 -1/3; -1/3 2/3 -1/3; -1/3 -1/3 2/3];

	if (strain'*J*strain<1e-10 || abs(i*strain)<1e-10)
		strain(1) = strain(1)+1e-10;
	end

	eta = 0.1*strain;
	dir = strain/norm(strain);

	F1 = ft/3*i*eta;
	dF1_deta = ft/3*i_col;
	dir1 = i'*i*strain; dir1 = dir1/norm(dir1);
 
	F2 = fs*sqrt(0.5)*sqrt(eta'*J*eta);
	dF2_deta = sqrt(0.5)*fs*J*eta/sqrt(eta'*J*eta);
	dir2 = J*strain; dir2 = dir2/norm(dir2);

	if F1<0
		dir = dir2;
	end

	if (modeId==1) %R1
		if (F1<0.0)
			dF1_deta=dF1_deta-ft*1e6*i_col;
		end

		Failure1 = dir1'*(dF1_deta-stress);
		Failure2 = dir2'*(dF2_deta-stress);
	elseif (modeId==2) %R2
		if (F1<0.0)
			dF_deta=F2*dF2_deta/sqrt(F2^2)-ft*1e6*i_col;
		else
			dF_deta = (F1*dF1_deta + F2*dF2_deta)/sqrt(F1^2+F2^2);
		end

		Failure1 = dir'*(dF_deta-stress);
		Failure2 = dir'*(dF_deta-stress);

	else %DP
		eRef = dp_eRef;

		p = i*strain;

		F = 0;
		dF_deta = 0;
		if p>0.0
			F       = F + sqrt(F1^2 + F2^2);
			dF_deta = dF_deta+ (F1*dF1_deta + F2*dF2_deta)/sqrt(F1^2+F2^2+1e-3);
		else
			F       = F       + F2*(1-p/eRef);
			dF_deta = dF_deta + dF2_deta*(1-p/eRef);
		end

		Failure1 = dir'*(dF_deta-stress);
		Failure2 = dir'*(dF_deta-stress);
	end
end
