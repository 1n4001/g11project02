% Reads in the training, validation, and test sets for group 11 project 1
% Performs a 5-fold least squaerror Regression on the data.
clear;
close all;
scale = 125;

input = csvread('C:\users\jeffrey\g11project02\Data\train.csv');
input(:,3) = ~input(:,2);

test = csvread('C:\users\jeffrey\g11project02\Data\test.csv');
test(:,3) = ~test(:,2);

train=zeros(8160,3,5);
valid=zeros(2040,3,5);
L_valid = zeros(2040,1,5);
beta = ones(2,1,5);
accuracy = zeros(5,1);

for i=1:5
    Minimum = (i-1)*2040;
    Maximum = i*2040;
    v_mask = logical(1:10200 > Minimum & 1:10200 <= Maximum);
    
    train(:,:,i) = input(~v_mask,:);
    valid(:,:,i) = input(v_mask,:);
end


for i=1:5
    beta(:,1,i) = irls(train(:,1,i), train(:,3,i), 1e-12, 50);
    
    L_valid(:,1,i) = logistic(beta(:,1,i),valid(:,1,i));
    cas_mask = logical(valid(:,3,i));
    reg_mask = logical(valid(:,2,i));
    
    false_cas = logical(cas_mask & L_valid(:,1,i)<0.5);
    false_reg = logical(reg_mask & L_valid(:,1,i)>=0.5);
    
    accuracy(i) = 1 - sum(false_cas + false_reg)/2040;
    
    figure(i)
    hold off
    semilogx(60*valid(:,1,i), valid(:,3,i),'b.');
    hold on
    semilogx(60*valid(false_cas,1,i), valid(false_cas,3,i),'r.');
    semilogx(60*valid(false_reg,1,i), valid(false_reg,3,i),'r.');    
    plot(60*(0.01:0.01:25),logistic(beta(:,1,i),0.01:0.01:25),'k');
    xlabel('Rental Duration in Minutes');
    ylabel('Probability of Being a Casual Rider');
    title( cat(2,'Validation Set ',num2str(i)) );

    xlim([0.6 1500]);
    set(gca,'XTickLabel',{'1','10','100','1,000'});

    grid on;
    
    location = [0.15 0.775 0.33 0.05];
    str = cat(2,'Model Accuracy: ', num2str(accuracy(i)*100), '%');
    annotation('textbox',location,'String',str,'BackgroundColor',[1 1 1]);

end

%% Plot test set and get accuracy

best = find(accuracy == max(accuracy));
test_size = length(test(:,1));

L_test = logistic(beta(:,1,best),test(:,1));
cas_mask = logical(test(:,3));
reg_mask = logical(test(:,2));

false_cas = logical(cas_mask & L_test < 0.5);
false_reg = logical(reg_mask & L_test >= 0.5);

final_accuracy = 1 - sum(false_cas + false_reg)/test_size;


figure(20)
hold off
semilogx(60*test(:,1), test(:,3),'b.');
hold on
semilogx(60*test(false_cas,1), test(false_cas,3),'r.');
semilogx(60*test(false_reg,1), test(false_reg,3),'r.');    
plot(60*(0.01:0.01:25),logistic(beta(:,1,best),0.01:0.01:25),'k');
xlabel('Rental Duration in Minutes');
ylabel('Probability of Being a Casual Rider');
title( 'Final Test Set' );

xlim([0.60 1500]);
set(gca,'XTickLabel',{'1','10','100','1,000'});

grid on;

location = [0.15 0.775 0.33 0.05];
str = cat(2,'Model Accuracy: ',num2str(final_accuracy*100), '%');
annotation('textbox',location,'String',str,'BackgroundColor',[1 1 1]);


%% Plot Histograms

% ave of the traing set = 17.59749 minutes (1.2455 on log scale)
% 50% mrk of sigmooid  = 41.774033 minutes (1.6209064 on log scale)

[hc xc]=hist(log10(60*test(cas_mask,1)),80);
[hr xr]=hist(log10(60*test(reg_mask,1)),100);
[hh xx]=hist(log10(60*test(:,1)),100);

% Logistic Models
figure(30)

h2 = axes;
bar(h2,xc,hc,'FaceColor','r')
set(h2,'Ydir','reverse','box','off','YAxisLocation','right','XAxisLocation','top','Ylim',[0 scale]);
set(h2,'XLim',[0 3],'XTickLabel',{});
ylabel(h2,'Number of Casual Riders');

Y = get(h2,'YLabel');
set(Y,'Rotation',-90, 'Position', get(Y, 'Position').*[1.025 1 1], 'color', 'r');

h1 = axes;
bar(h1,xr,hr)
hold(h1,'on')
plot(h1,xx,scale*logistic(beta(:,1,best),10.^xx / 60),'color', [0.15 0.75 0.05],'LineWidth',2);
plot(h1,[1.6209064 1.6209064],[0 scale],'--k','LineWidth',2);

set(h1,'color','none','Ylim',[0 scale], 'box', 'off');
set(h1,'XLim',[0 3],'XTickLabel',{'1','','10','','100','','1,000'});
xlabel(h1,'Rental Duration in Minutes');
ylabel(h1,'Number of Registered Riders','color','b');

location = [0.565 0.33 0.33 0.05];
annotation('textbox',location,'String',str,'BackgroundColor',[1 1 1]);
title('Logistic Model Prediction of Duration Threshold');



% Naive Model
figure(40)

h2 = axes;
bar(h2,xc,hc,'FaceColor','r')
set(h2,'Ydir','reverse','box','off','YAxisLocation','right','XAxisLocation','top','Ylim',[0 scale]);
set(h2,'XLim',[0 3],'XTickLabel',{});
ylabel(h2,'Number of Casual Riders');

Y = get(h2,'YLabel');
set(Y,'Rotation',-90, 'Position', get(Y, 'Position').*[1.025 1 1], 'color', 'r');

h1 = axes;
bar(h1,xr,hr)
hold(h1,'on')
plot(h1,[1.2455 1.2455],[0 scale],'--k','LineWidth',2);

set(h1,'color','none','Ylim',[0 scale], 'box', 'off');
set(h1,'XLim',[0 3],'XTickLabel',{'1','','10','','100','','1,000'});
xlabel(h1,'Rental Duration in Minutes');
ylabel(h1,'Number of Registered Riders','color', 'b');

location = [0.5 0.65 0.33 0.05];
str = cat(2,'Model Accuracy: ',num2str(75.7148), '%');
annotation('textbox',location,'String',str,'BackgroundColor',[1 1 1]);
title('Naive Prediction of Duration Threshold');



% Both Models
figure(50)

h2 = axes;
bar(h2,xc,hc,'FaceColor','r')
set(h2,'Ydir','reverse','box','off','YAxisLocation','right','XAxisLocation','top','Ylim',[0 scale]);
set(h2,'XLim',[0 3],'XTickLabel',{});
ylabel(h2,'Number of Casual Riders');

Y = get(h2,'YLabel');
set(Y,'Rotation',-90, 'Position', get(Y, 'Position').*[1.025 1 1], 'color', 'r');

h1 = axes;
bar(h1,xr,hr)
hold(h1,'on')
plot(h1,[1.2455 1.2455],[0 scale],'--k','LineWidth',2);
plot(h1,xx,scale*logistic(beta(:,1,best),10.^xx / 60),'color', [0.15 0.75 0.05],'LineWidth',2);
plot(h1,[1.6209064 1.6209064],[0 scale],'--k','LineWidth',2);

set(h1,'color','none','Ylim',[0 scale], 'box', 'off');
set(h1,'XLim',[0 3],'XTickLabel',{'1','','10','','100','','1,000'});
xlabel(h1,'Rental Duration in Minutes');
ylabel(h1,'Number of Registered Riders','color', 'b');

title('Both Predictions of Duration Threshold');


