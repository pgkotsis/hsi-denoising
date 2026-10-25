load("/home/panos/Desktop/HSI/Project/data/DC/DC1.mat", "dc");

I1 = double(dc);
y = 51;
x = 601;
I1 = I1(x:x+199, y:y+199, :);
I1 = I1 - repmat(min(min(I1, [], 1), [], 2), [200 200]);
I1 = I1 ./ repmat(max(max(I1, [], 1), [], 2), [200 200]);
image = I1;

image_noisy = image + randn(size(I1)) * (100/255);
image_noisy(image_noisy < 0) = 0;
image_noisy(image_noisy > 1) = 1;
%save('/home/panos/Desktop/HSI/new/deep-hs-prior/data/denoising/dc/denoising.mat', 'image', 'image_noisy');
%imshow(cat(3, I1(:,:,57), I1(:,:,27), I1(:,:,17)))

%%
load("/home/panos/Desktop/HSI/new/deep-hs-prior/results/dc/result_denoising_2D_it01800.mat", "pred_avg");
p = psnr(double(pred_avg), image(:, :, 1:191));
s = ssim(double(pred_avg), image(:, :, 1:191));

fprintf('psnr: %f, ssim: %f  \n', p, s);
figure; imshow(cat(3, pred_avg(:, :, 57), pred_avg(:, :, 27), pred_avg(:, :, 17)));

%%
imwrite(cat(3, pred_avg(:, :, 57), pred_avg(:, :, 27), pred_avg(:, :, 17)), 'dc.png');