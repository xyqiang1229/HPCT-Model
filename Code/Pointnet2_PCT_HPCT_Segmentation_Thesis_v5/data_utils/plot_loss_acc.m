% ע��ԭʼtxt�ļ���ͨ��excel����һ�£��ָ���ѡ"-"��ȥ�����ں���ʾ������ȡtxt�ļ�
filename = 'F:\HPC\Research\PCSS\Predict\Own_Thesis\Own_Thesis_v2_MultiSource\PCT_230329_1\logs\PCT_sem_seg_pre.txt';
fileID = fopen(filename, 'r');

% ��ʼ������
epochs = 50;
training_mean_loss = zeros(epochs, 1);
training_accuracy = zeros(epochs, 1);
eval_point_accuracy = zeros(epochs, 1);

% ���ж�ȡ�ͽ���
epoch = 0;
while ~feof(fileID)
    tline = fgetl(fileID);
    if contains(tline, '**** Epoch')
        epoch = sscanf(tline, ' **** Epoch %d (1/50) ****');
    elseif contains(tline, 'Training mean loss')
        training_mean_loss(epoch) = sscanf(tline, ' Training mean loss: %f');
    elseif contains(tline, 'Training accuracy')
        training_accuracy(epoch) = sscanf(tline, ' Training accuracy: %f');
    elseif contains(tline, 'eval point accuracy')
        eval_point_accuracy(epoch) = sscanf(tline, ' eval point accuracy: %f');
    end
end

% ����˫������ͼ
figure
yyaxis left
plot(1:epochs, training_mean_loss, 'r-', 'LineWidth', 1.5)
ylabel('Training Loss')
ylim([0 max(training_mean_loss)*1.2])

yyaxis right
plot(1:epochs, training_accuracy * 100, 'b-', 'LineWidth', 1.5)
hold on
% plot(1:epochs, eval_point_accuracy * 100, 'g-', 'LineWidth', 1.5)
% ylabel('Training Accuracy & Eval Accuracy')
ylabel('Training Accuracy')
ylim([98 100])
ytickformat('percentage')

xlabel('Epoch')
% legend('Training Loss', 'Training Accuracy', 'Eval Accuracy')
legend('Training Loss', 'Training Accuracy')
grid on
