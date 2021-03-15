
function [OA,Kappa,producerA] = CalAccuracy(predict_label,label)

k = 0;
for i = 1:size(label)
    if(label(i) == predict_label(i))
        k = k+1;
    end
end
n = length(label);
OA = k / n;

for i= 1:max(label(:))
    correct_sum(i) = sum(label(find(predict_label==i))==i);
    reali(i) = sum(label==i);
    predicti(i) = sum(predict_label==i);
    producerA(i) = correct_sum(i) / reali(i);
end

Kappa = (n*sum(correct_sum) - sum(reali .* predicti)) / (n*n - sum(reali .* predicti));

end