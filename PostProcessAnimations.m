close all
clear all
clc

PostProcess = [false, true, false, false, false, true];
SaveAnimations = [false, false, false, false, false, true];
FinalFiguresOnly = [true, true, true, true, true, true];
videoRes = 300;

E = 200e9;
v = 0.3;
K = E/(3*(1-2*v));
G = E/(2*(1-v));
ft = 150e6;
fs = 150e6;

%% 1: Mesh Refinement PlateWithHole
if PostProcess(1)
	colors = distinguishable_colors(8);
	marks = {'x','v','o','<','*','square','diamond','^'};
	i=0;
	for dx=["0.5","0.75","1.0","2.0","3.0","4.0","5.0","7.5"]
		i=i+1;
		Fldr = "Geometry_PlateWithHole/dir_-1/Gc_100000.0_dxRef_"+dx;
		Fldr2= "Geometry_PlateWithHole/dir_1/Gc_100000.0_dxRef_"+dx;
		saveFolder = "Animations/"+"MeshRef/"+dx+"_down";
		saveFolder2 = "Animations/"+"MeshRef/"+dx+"_up";
		savePrefix = "DOWN_dx"+string(dx)+"_";
		savePrefix2= "UP_dx"+string(dx)+"_";
		PlotLabel = "$\ell/\mathrm{dx}="+dx+"$";
		[a,b,c] = mkdir(saveFolder);
		[a,b,c] = mkdir(saveFolder2);
	
		Exists = true;
		try
			T_Data = h5read(Fldr+"/time_data.hdf5", "/data");
			T_Data2= h5read(Fldr2+"/time_data.hdf5", "/data");
		catch
			% skip cases that haven't ran yet
			Exists = false;
		end
		if Exists
			%T_DataNames = h5read(Fldr+"/time_data.hdf5", "/data_names");
			F = [0, T_Data(4,:)];
			u = [0, T_Data(6,:)];
			figure(501)
			l(2*i-1) = plot(u*1e3,F*1e-6, 'DisplayName',PlotLabel,'Color',colors(i,:),'LineWidth',1);
			mrks{2*i-1} = marks{i};
			hold on
	
			F = [0, T_Data2(4,:)];
			u = [0, T_Data2(6,:)];
			l(2*i) = plot(u*1e3,F*1e-6, 'DisplayName',PlotLabel,'Color',colors(i,:),'LineWidth',1,'HandleVisibility','off');
			mrks{2*i} = marks{i};
		
			if SaveAnimations(1)
				MakeAnimations(Fldr, saveFolder, savePrefix, videoRes, -3e-3, FinalFiguresOnly(1),"y");
				MakeAnimations(Fldr2, saveFolder2, savePrefix2, videoRes, 3e-3, FinalFiguresOnly(1),"y");
			end
		end
	end
	fMeshRef = figure(501);
	xline(0,'LineWidth',0.1,'HandleVisibility','off','Color',[.7 .7 .7])
	yline(0,'LineWidth',0.1,'HandleVisibility','off','Color',[.7 .7 .7])
	xlabel('$\varepsilon_{yy}\;[-]\;\;\; \times10^{-3}$', 'Interpreter','latex');
	ylabel('$\sigma_{yy}\;[\mathrm{MPa}]$', 'Interpreter','latex');
	leg = legend('Location','eastoutside', 'Interpreter','latex');
	xlim([-3,3])
	ylim([-120, 95])
	grid off
	mrkrs = plotsparsemarkers(l, leg, mrks);
	for i=1:length(mrkrs)
		mrkrs(i).MarkerSize = 6;
	end
	saveFigNow(fMeshRef, "Meshref_LoadDisp", 6, true, false, 0)
end

%% 2: Variations in Gc PlateWithHole
if PostProcess(2)
	i=0;
	colors = distinguishable_colors(9);
	marks = {'x','|','o','+','*','square','diamond','^','x'};
	for Gc=[1e4, 2.5e4, 5e4, 7.5e4, 1e5, 2.5e5, 5e5, 7.5e5, 1e6]
		l_ch(1) = K*Gc/ft^2;
		l_ch(2) = 2*G*Gc/fs^2;
		msg = "Gc="+string(Gc/1000)+" kJ/m2, l1="+string(l_ch(1))+", l2="+string(l_ch(2));
		fprintf(msg+"\n")
	
		i=i+1;
		Fldr = "Geometry_PlateWithHole/dir_-1/Gc_"+string(Gc)+".0_dxRef_3.0";
		Fldr2= "Geometry_PlateWithHole/dir_1/Gc_"+string(Gc)+".0_dxRef_3.0";
		saveFolder = "Animations/"+"Gc/"+string(Gc)+"_down";
		saveFolder2 = "Animations/"+"Gc/"+string(Gc)+"_up";
		savePrefix = "DOWN_Gc"+string(Gc)+"_";
		savePrefix2= "UP_Gc"+string(Gc)+"_";
		PlotLabel = "$G_\mathrm{c}="+string(Gc/1000)+"\;\mathrm{kJ}/\mathrm{m}^2$";
		[a,b,c] = mkdir(saveFolder);
		[a,b,c] = mkdir(saveFolder2);
	
		Exists = true;
		try
			T_Data = h5read(Fldr+"/time_data.hdf5", "/data");
			T_Data2= h5read(Fldr2+"/time_data.hdf5", "/data");
		catch
			%skip cases that haven't ran yet
			Exists = false;
		end
		if Exists
			%T_DataNames = h5read(Fldr+"/time_data.hdf5", "/data_names");
			F = [0, T_Data2(4,:)];
			u = [0, T_Data2(6,:)];
			%F(F<0) = 0.0;
	
			figure(502)
			lType = "-";
			l(2*i-1) = plot(u*1e3,F*1e-6, lType, 'DisplayName',PlotLabel,'Color',colors(i,:),'LineWidth',1);
			hold on
			mrks{2*i-1} = marks{i};
	
			F = [0, T_Data(4,:)];
			u = [0, T_Data(6,:)];
			%F(F>0) = 0.0;
	
			if Gc<5e4
				lType = "-.";
			else
				lType = "-";
			end
			l(2*i) = plot(u*1e3,F*1e-6, lType, 'DisplayName',PlotLabel,'Color',colors(i,:),'LineWidth',1,'HandleVisibility','off');
			mrks{2*i} = marks{i};
		
			if SaveAnimations(2)
				MakeAnimations(Fldr, saveFolder, savePrefix, videoRes, false, FinalFiguresOnly(2),"y");
				MakeAnimations(Fldr2, saveFolder2, savePrefix2, videoRes, false, FinalFiguresOnly(2),"y");
			end
		end
	end
	fMeshRef = figure(502);
	xline(0,'LineWidth',0.1,'HandleVisibility','off','Color',[.7 .7 .7])
	yline(0,'LineWidth',0.1,'HandleVisibility','off','Color',[.7 .7 .7])
	xlabel('$\varepsilon_{yy}\;[-]\;\;\; \times10^{-3}$', 'Interpreter','latex');
	ylabel('$\sigma_{yy}\;[\mathrm{MPa}]$', 'Interpreter','latex');
	leg = legend('Location','eastoutside', 'Interpreter','latex');
	xlim([-4,4])
	%ylim([-120, 95])
	grid off
	mrkrs = plotsparsemarkers(l, leg, mrks);
	for i=1:length(mrkrs)
		mrkrs(i).MarkerSize = 6;
	end
	saveFigNow(fMeshRef, "Gc_LoadDisp", 6, true, false, 0)
end

%% Variations in l
if PostProcess(3)
	i=0;
	colors = distinguishable_colors(10);
	marks = {'x','|','o','+','*','square','diamond','^','x'};
	for ell=[0.1, 0.075, 0.05, 0.025, 0.01]
		i=i+1;
		Fldr = "Geometry_PlateWithHole_LSweep/dir_-1/ell_"+string(ell);
		Fldr2= "Geometry_PlateWithHole_LSweep/dir_1/ell_"+string(ell);
		saveFolder = "Animations/"+"Plate_LRef/"+string(ell)+"_down";
		saveFolder2 = "Animations/"+"Plate_LRef/"+string(ell)+"_up";
		savePrefix = "DOWN_ell"+string(ell)+"_";
		savePrefix2= "UP_ell"+string(ell)+"_";
		PlotLabel = "$\ell="+string(ell)+"\;\mathrm{m}$";
		[a,b,c] = mkdir(saveFolder);
		[a,b,c] = mkdir(saveFolder2);
	
		Exists = true;
		try
			T_Data = h5read(Fldr+"/time_data.hdf5", "/data");
			T_Data2= h5read(Fldr2+"/time_data.hdf5", "/data");
		catch
			%skip cases that haven't ran yet
			Exists = false;
		end
		if Exists
			%T_DataNames = h5read(Fldr+"/time_data.hdf5", "/data_names");
			F = [0, T_Data2(4,:)];
			u = [0, T_Data2(6,:)];
			%F(F<0) = 0.0;
	
			figure(512)
			lType = "-";
			l(2*i-1) = plot(u*1e3,F*1e-6, lType, 'DisplayName',PlotLabel,'Color',colors(i,:),'LineWidth',1);
			hold on
			mrks{2*i-1} = marks{i};
	
			F = [0, T_Data(4,:)];
			u = [0, T_Data(6,:)];
			%F(F>0) = 0.0;
	
			l(2*i) = plot(u*1e3,F*1e-6, lType, 'DisplayName',PlotLabel,'Color',colors(i,:),'LineWidth',1,'HandleVisibility','off');
			mrks{2*i} = marks{i};
		
			if SaveAnimations(3)
				MakeAnimations(Fldr, saveFolder, savePrefix, videoRes, -3e-3, FinalFiguresOnly(3),"y");
				MakeAnimations(Fldr2, saveFolder2, savePrefix2, videoRes, 3e-3, FinalFiguresOnly(3),"y");
			end
		end
	end
	fMeshRef = figure(512);
	xline(0,'LineWidth',0.1,'HandleVisibility','off','Color',[.7 .7 .7])
	yline(0,'LineWidth',0.1,'HandleVisibility','off','Color',[.7 .7 .7])
	xlabel('$\varepsilon_{yy}\;[-]\;\;\; \times10^{-3}$', 'Interpreter','latex');
	ylabel('$\sigma_{yy}\;[\mathrm{MPa}]$', 'Interpreter','latex');
	leg = legend('Location','eastoutside', 'Interpreter','latex');
	xlim([-3,3])
	ylim([-120, 95])
	grid off
	mrkrs = plotsparsemarkers(l, leg, mrks);
	for i=1:length(mrkrs)
		mrkrs(i).MarkerSize = 6;
	end
	saveFigNow(fMeshRef, "ellRef_LoadDisp", 6, true, false, 0)

	% delete(leg)
	% xlim([-0.65, -0.45])
	% ylim([-120, -90])
	% saveFigNow(fMeshRef, "ellRef_LoadDisp_ZoomCompress", 4, false, false, 0)
	% 
	% xlim([0.35, 0.55])
	% ylim([70, 90])
	% saveFigNow(fMeshRef, "ellRef_LoadDisp_ZoomTension", 4, false, false, 0)
end

%% 4: Variations in failure surf + Gc SENT
if PostProcess(4)
	j=0;
	colors = distinguishable_colors(9);
	marks = {'x','|','o','+','*','square','diamond','^','x'};
	for Gc=[1e5, 1e4]
		j=j+1;
		figure(503+2*j)
		figure(504+2*j)
		i=0;
        dpNames = {"10^{-5}","5\cdot 10^{-5}", "10^{-4}","5\cdot 10^{-4}", "10^{-3}","10^{-2}", "10^{-1}","1"};
		for DP=[0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 1e-1, 1e0]
			if DP==0
				failure = "r1";
				DPName = "1.0";
				PlotLabel = failure;
			else
				failure = "DP";
				DPName = string(DP);
				PlotLabel = failure+", $\varepsilon_\mathrm{ref}="+dpNames(i)+"$";
			end
			i=i+1;
			%Fldr = "Geometry_SENT/dir_1/Gc_"+string(Gc)+".0_FailureMode_"+failure+"_DPref_"+DPName;
			Fldr2 = "Geometry_SENT/dir_-1/Gc_"+string(Gc)+".0_FailureMode_"+failure+"_DPref_"+DPName;
	
			%saveFolder = "Animations/SENT/"+"Gc_"+string(Gc)+failure+"_DPref_"+DPName+"_down";
			saveFolder2 = "Animations/SENT2/"+"Gc_"+string(Gc)+failure+"_DPref_"+DPName+"_up";
			%savePrefix = "UP_Gc"+string(Gc)+"_"+failure+DPName+"_";
			savePrefix2= "RIGHT_Gc"+string(Gc)+"_"+failure+DPName+"_";
	
			%[a,b,c] = mkdir(saveFolder);
			[a,b,c] = mkdir(saveFolder2);
	
			Exists = true;
			try
				%T_Data = h5read(Fldr+"/time_data.hdf5", "/data");
				T_Data2= h5read(Fldr2+"/time_data.hdf5", "/data");
			catch
				%skip cases that haven't ran yet
				Exists = false;
			end
	
			if Exists
				%T_DataNames = h5read(Fldr+"/time_data.hdf5", "/data_names");
				F = [0, T_Data2(4,:)];
				u = [0, T_Data2(6,:)];
				%F(F<0) = 0.0;

				figure(503+2*j)
				lType = "-";
				l(i) = plot(u*1e3,F*1e-6, lType, 'DisplayName',PlotLabel,'Color',colors(i,:),'LineWidth',1);
				hold on
				mrks{i} = marks{i};
				% 
				%F = [0, T_Data(4,:)];
				%u = [0, T_Data(6,:)];
				%F(F>0) = 0.0;
		
				% figure(504+2*j)
				% l2(i) = plot(u*1e3,F*1e-6, lType, 'DisplayName',PlotLabel,'Color',colors(i,:),'LineWidth',1);
				% hold on
				% mrks2{i} = marks{i};
			
				if SaveAnimations(4)
					%MakeAnimations(Fldr, saveFolder, savePrefix, videoRes, false, FinalFiguresOnly(4),"x");
					MakeAnimations(Fldr2, saveFolder2, savePrefix2, videoRes, false, FinalFiguresOnly(4),"x");
				end
			end
		end
	
		fDP = figure(503+2*j);
		xlabel('$\varepsilon_{xx}\;[-]\;\;\; \times10^{-3}$', 'Interpreter','latex');
		ylabel('$\sigma_{xx}\;[\mathrm{MPa}]$', 'Interpreter','latex');
		leg = legend('Location','eastoutside', 'Interpreter','latex');
		xlim([0,3])
		%ylim([-120, 95])
		grid off
		mrkrs = plotsparsemarkers(l, leg, mrks);
		for i=1:length(mrkrs)
			mrkrs(i).MarkerSize = 6;
		end
		saveFigNow(fDP, "SENT_LoadDisp_Gc"+string(Gc), 6, true, false, 0)
	
		% fDP2 = figure(504+2*j);
		% xlabel('$\varepsilon_{yy}\;[-]\;\;\; \times10^{-3}$', 'Interpreter','latex');
		% ylabel('$\sigma_{yy}\;[\mathrm{MPa}]$', 'Interpreter','latex');
		% leg = legend('Location','eastoutside', 'Interpreter','latex');
		% %xlim([-4,4])
		% %ylim([-120, 95])
		% grid off
		% mrkrs = plotsparsemarkers(l2, leg, mrks2);
		% for i=1:length(mrkrs)
		% 	mrkrs(i).MarkerSize = 6;
		% end
		%saveFigNow(fDP, "Gc_LoadDisp", 6, true, false, 0)
	end
end

%% 5: Dynamic
if PostProcess(5)
	i=0;
	colors = distinguishable_colors(9);
	marks = {'x','|','o','+','*','square','diamond','^','x'};
	for Gc=[1e3, 2.5e3, 5e3, 7.5e3, 1e4, 2.5e4, 5e4, 7.5e4, 1e5]
		i=i+1;
		Fldr = "Geometry_Dynamic/Gc_"+string(Gc)+".0";
		saveFolder = "Animations/Dynamic/"+"Gc_"+string(Gc);
		savePrefix = "Dynamic_Gc"+string(Gc)+"_";
		PlotLabel = "$G_\mathrm{c}="+string(Gc/1000)+"\;\mathrm{kJ}/\mathrm{m}^2$";
		[a,b,c] = mkdir(saveFolder);
	
		Exists = true;
		try
			T_Data = h5read(Fldr+"/time_data.hdf5", "/data");
		catch
			%skip cases that haven't ran yet
			Exists = false;
		end
		if Exists
			T_DataNames = h5read(Fldr+"/time_data.hdf5", "/data_names");
			T = [0, T_Data(1,:)];
			L = [0, T_Data(3,:)];
			%F(F<0) = 0.0;
	
			figure(507)
			lType = "-";
			l(i) = plot(T*1000,L, lType, 'DisplayName',PlotLabel,'Color',colors(i,:),'LineWidth',1);
			hold on
			mrks{i} = marks{i};
		
			if SaveAnimations(5)
				MakeAnimations(Fldr, saveFolder, savePrefix, videoRes, 0.8e-3, FinalFiguresOnly(5),"D");
			end
		end
	end
	fMeshRef = figure(507);
	xlabel('$t\;[\mathrm{ms}]$', 'Interpreter','latex');
	ylabel('$L_\mathrm{cracks}\;[\mathrm{m}]$', 'Interpreter','latex');
	leg = legend('Location','eastoutside', 'Interpreter','latex');
	xlim([0,0.8])
	ylim([0, 1.0])
	grid off
	mrkrs = plotsparsemarkers(l, leg, mrks);
	for i=1:length(mrkrs)
		mrkrs(i).MarkerSize = 6;
	end
	saveFigNow(fMeshRef, "Gc_Dynamic", 6, true, false, 0)
end


%% 6: Limitations, plate with hole length scale sweep
if PostProcess(6)
	i=0;
	colors = distinguishable_colors(9);
	marks = {'x','|','o','+','*','square','diamond','^','x'};
	for ell=[0.1, 0.075, 0.05, 0.025, 0.01]
	
		i=i+1;
		Fldr = "Geometry_PlateWithHole/ellref_GC_10000.0/ell_"+string(ell);
		saveFolder = "Animations/"+"FailedCases/"+string(ell);
		savePrefix = "LIMITATIONS_Ell"+string(ell)+"_";
		PlotLabel = "$\ell="+string(ell)+"\;\mathrm{m}$";
		[a,b,c] = mkdir(saveFolder);
	
		Exists = true;
		try
			T_Data = h5read(Fldr+"/time_data.hdf5", "/data");
		catch
			%skip cases that haven't ran yet
			Exists = false;
		end
		if Exists
			figure(548)
			hold on
			mrks{i} = marks{i};
	
			F = [0, T_Data(4,:)];
			u = [0, T_Data(6,:)];
	
			lType = "-.";
			l(i) = plot(u*1e3,F*1e-6, lType, 'DisplayName',PlotLabel,'Color',colors(i,:),'LineWidth',1,'HandleVisibility','off');
			mrks{i} = marks{i};
		
			if SaveAnimations(6)
				MakeAnimations(Fldr, saveFolder, savePrefix, videoRes, false, FinalFiguresOnly(6),"y");
			end
		end
	end
	fMeshRef = figure(502);
	xlabel('$\varepsilon_{yy}\;[-]\;\;\; \times10^{-3}$', 'Interpreter','latex');
	ylabel('$\sigma_{yy}\;[\mathrm{MPa}]$', 'Interpreter','latex');
	leg = legend('Location','eastoutside', 'Interpreter','latex');
	xlim([-4,0])
	ylim([-70, 0])
	grid off
	mrkrs = plotsparsemarkers(l, leg, mrks);
	for i=1:length(mrkrs)
		mrkrs(i).MarkerSize = 6;
	end
	saveFigNow(fMeshRef, "Limitations_LoadDisp", 6, true, false, 0)
end








function MakeAnimations(input, output, sprefix, videoRes, e_max, saveFinalOnly, loadDir)
	if nargin < 6
		saveFinalOnly = false;
    end
    if nargin < 7
        loadDir = "y";
    end

	Fields = ["u_x", "u_y", "phasefield"];
	Field_Labels = {{}, {"$u_y$","$[\mathrm{mm}]$"}, {"$\phi$","$[-]$"}};
	D_scale = [1000, 1000, 1];
	DefScale = [10, 10, 0];
	T_Data = h5read(input+"/time_data.hdf5", "/data");
	T_DataNames = h5read(input+"/time_data.hdf5", "/data_names");
	nmax = size(T_Data, 2);
	if isfloat(e_max)
		if loadDir ~= "D"
			[~, nmax] = min(abs(T_Data(6,:)-e_max));
		else
			[~, nmax] = min(abs(T_Data(1,:)-e_max));
		end
	end

	fprintf('\n=== MakeAnimations: %s (%d frames) ===\n', output, nmax);

	%% Load Displacement
	if loadDir ~= "D"
		fg{1} = figure('Visible','off');
		fg{1}.Units = "centimeters";
		fg{1}.Position(3) = 6;
		fg{1}.Position(4) = 6;
		set(fg{1},'color','w');
		e = [0, T_Data(6,1:nmax)]*1e3;
		F = [0, T_Data(4,1:nmax)]*1e-6;
		if ~saveFinalOnly
			vidfile{1} = VideoWriter(output+"/TimeData.mp4",'MPEG-4');
			vidfile{1}.FrameRate = 20;
			open(vidfile{1});
			hLine = plot(e(1), F(1), '-k','LineWidth',2);
		else
			plot(e, F, '-k','LineWidth',2);
		end
		xlabel('Displacement [mm]');
		ylabel('Applied Stress (MPa)');
		xlim([min(e), max(e)])
		ylim([min(F), max(F)])
		if ~saveFinalOnly
			fprintf('  Load-displacement video: ');
			for i=1:1:nmax
				set(hLine, 'XData', e(1:i+1), 'YData', F(1:i+1));
				drawnow();
				writeHighResFrame(vidfile{1}, fg{1}, videoRes);
				if i > 1, fprintf(repmat('\b',1,11)); end
				fprintf('%4d / %4d', i, nmax);
			end
			fprintf(' done\n');
		end
		fg{1}.Visible = 'on';
		saveFigNow(fg{1},output+"/"+sprefix+"LoadDisp", 8, false, false, 0);
		if ~saveFinalOnly
			close(vidfile{1});
		end
		close(fg{1});
	end

	X_Mesh = h5read(input+"/Mesh.hdf5", "/Mesh_DG2/X");
	Y_Mesh = h5read(input+"/Mesh.hdf5", "/Mesh_DG2/Y");
	%Z_Mesh = h5read(Folder+"/Mesh.hdf5", "/Mesh/Z")
	xRange = [min(X_Mesh,[],"all"), max(X_Mesh,[],"all")];
	xRange(1) = xRange(1) - 0.1*(xRange(2)-xRange(1));
	xRange(2) = xRange(2) + 0.1*(xRange(2)-xRange(1));
	yRange = [min(Y_Mesh,[],"all"), max(Y_Mesh,[],"all")];
	yRange(1) = yRange(1) - 0.1*(yRange(2)-yRange(1));
	yRange(2) = yRange(2) + 0.1*(yRange(2)-yRange(1));
	
	order = [1 6 2 4 3 5];

	% Create figures, patches, colorbars and video writers (once)
	for j=1:length(Fields)
		fg{j} = figure('Visible','off');
		fg{j}.Units = "centimeters";
		fg{j}.Position(3) = 10;
		fg{j}.Position(4) = 10;
		set(fg{j},'color','w');

		if ~saveFinalOnly
			vidfile{j} = VideoWriter(output+"/"+Fields{j}+".mp4",'MPEG-4');
			vidfile{j}.FrameRate = 20;
			open(vidfile{j});
		end

		axis off
		axis equal
		xlim(xRange)
		ylim(yRange)
		hold on

		% Create initial patch with dummy data
		initData = zeros(size(X_Mesh));
		hPatch{j} = plotSurf(X_Mesh, Y_Mesh, initData, j<=2);

		
		if j<=2
			if (loadDir ~= "D")
				hRect1{j} = rectangle('Position',[-0.1, 1, 1.2, 0.1], ...
					'FaceColor',[0.5 0.5 0.5],'EdgeColor','none');
				hRect2{j} = rectangle('Position',[-0.1, -0.1, 1.2, 0.1], ...
					'FaceColor',[0.5 0.5 0.5],'EdgeColor','none');
			end
		else
			clim([0,1])
			colormap hot
		end

		cb = colorbar;
		if j==1
			cb.Title.String = "$u_x\;[\mathrm{mm}]$";
		elseif j==2
			cb.Title.String = "$u_y\;[\mathrm{mm}]$";
		else
			cb.Title.String = "$\phi\;[-]$";
		end
		cb.Title.Interpreter = 'latex';
		hTitle{j} = title('');
	end

	% Animation loop: update existing graphics objects each frame
	if ~saveFinalOnly
		fprintf('  Field animations:         ');
		for i=1:1:nmax
			if i > 1, fprintf(repmat('\b',1,11)); end
			fprintf('%4d / %4d', i, nmax);
			t = T_Data(1,i);
			if (loadDir ~= "D")
				Uext = T_Data(6,i);
			end
	
			dataFile = input + "/Outputs_"+string(i)+".hdf5";
			for j=1:length(Fields)
				D{j} = h5read(dataFile, "/Fields/"+Fields(j));
			end
	
			for j=1:length(Fields)
				if (DefScale(j)>0)
					X = X_Mesh + DefScale(j)*D{1};
					Y = Y_Mesh + DefScale(j)*D{2};
				else
					X = X_Mesh;
					Y = Y_Mesh;
				end
	
				% Update patch vertices and colour data
				set(hPatch{j}, 'XData', X(order,:), 'YData', Y(order,:), ...
					'CData', D{j}(order,:)*D_scale(j));
	
				if (loadDir ~= "D")
					if j<=2
                    	if loadDir=="y"
					    	set(hRect1{j}, 'Position', [-0.1, 1+Uext*DefScale(j), 1.2, 0.1]);
                    	else
                        	set(hRect1{j}, 'Position', [-0.1+Uext*DefScale(j), 1, 1.2, 0.1]);
                    	end
					end
				end
	
				if (i<nmax)
					set(hTitle{j}, 'String', ["t= "+string(t)+" s";Fields{j}]);
				else
					set(hTitle{j}, 'String', '');
				end
	
				drawnow();
				writeHighResFrame(vidfile{j}, fg{j}, videoRes);
			end
		end
		fprintf(' done\n');
	else
		i = nmax;
		if (loadDir ~= "D")
			Uext = T_Data(6,i);
		end
		dataFile = input + "/Outputs_"+string(i)+".hdf5";
		for j=1:length(Fields)
			D{j} = h5read(dataFile, "/Fields/"+Fields(j));
		end
		for j=1:length(Fields)
			if (DefScale(j)>0)
				X = X_Mesh + DefScale(j)*D{1};
				Y = Y_Mesh + DefScale(j)*D{2};
			else
				X = X_Mesh;
				Y = Y_Mesh;
			end
			set(hPatch{j}, 'XData', X(order,:), 'YData', Y(order,:), ...
				'CData', D{j}(order,:)*D_scale(j));

			if (loadDir ~= "D")
				if j<=2
                	if loadDir == "y"
				    	set(hRect1{j}, 'Position', [-0.1, 1+Uext*DefScale(j), 1.2, 0.1]);
                	else
                    	set(hRect1{j}, 'Position', [-0.1+Uext*DefScale(j), 1, 1.2, 0.1]);
                	end
				end
			end
			set(hTitle{j}, 'String', '');
			drawnow();
		end
	end

	for j=1:length(Fields)
		fg{j}.Visible = 'on';
		saveFigNow(fg{j}, output+"/"+sprefix+Fields{j}, 6, false, false, 0)
		if ~saveFinalOnly
			close(vidfile{j});
		end
		close(fg{j});
		if ~saveFinalOnly
			fprintf('  Saved %s video\n', Fields{j});
		else
			fprintf('  Saved %s final figure\n', Fields{j});
		end
	end
	fprintf('=== Done: %s ===\n', output);


end

function writeHighResFrame(vidWriter, fig, dpi)
	% Renders the figure at the specified DPI directly to an RGB
	% array in memory (no temp file), then writes it as a video frame.
	img = print(fig, '-RGBImage', ['-r', num2str(dpi)]);
	writeVideo(vidWriter, img);
end

function h = plotSurf(X, Y, Data, plotMesh)
	order = [1 6 2 4 3 5];

	% Vectorised reordering: each column becomes one patch face
	X_el = X(order,:);
	Y_el = Y(order,:);
	Z_el = zeros(size(X_el));
	Data_el = Data(order,:);

	if plotMesh
		h = patch(X_el, Y_el, Z_el, Data_el, 'EdgeColor','interp','FaceColor','k');
	else
		h = patch(X_el, Y_el, Z_el, Data_el, 'EdgeColor','none','FaceColor','interp');
	end
end

function saveFigNow(fg, sname, HFig, WFig, hasColorbar, cb)
	figure(fg);
	fprintf(sname+"  ")
	ax = gca;
	ax.FontSize = 8;
	fg.Units = 'centimeters';
	if (WFig==true || WFig==false)
		if (WFig)
			fg.Position = [2 2 16 HFig];
		else
			fg.Position = [2 2 8 HFig];
		end
	else
		fg.Position = [2 2 WFig HFig];
	end
	if (hasColorbar)
		cb.Position(1) = 0.85;
		cb.Position(2) = 0.1;
		cb.Position(4) = 0.65;
		if(HFig==5)
			ax.Position(1) = 0.10;
			ax.Position(2) = 0.20;
			ax.Position(3) = 0.70;
			ax.Position(4) = 0.70;
		else
			ax.Position(1) = 0.05;
			ax.Position(2) = 0.05;
			ax.Position(3) = 0.75;
			ax.Position(4) = 0.85;
		end
	end
	set(fg,'color','w');

	drawnow();
	print(fg, sname+".png",'-dpng','-r1200'); fprintf(".png  ")
	print(fg, sname+".jpg",'-djpeg','-r1200'); fprintf(".jpg  ")
	print(fg, sname+".eps",'-depsc','-r1200'); fprintf(".eps  ")
	print(fg, sname+".svg",'-dsvg','-r1200'); fprintf(".svg  ")
	print(fg, sname+".emf",'-dmeta','-r1200'); fprintf(".emf\n")
end

