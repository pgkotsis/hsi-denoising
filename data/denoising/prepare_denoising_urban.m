load("/home/panos/Desktop/HSI/Project/data/Urban/Urban_R162.mat", "Y");

I1 = double(Y);
I1 = reshape(I1, [162, 307, 307]);
I1 = permute(I1, [2, 3, 1]);
I1 = I1 - repmat(min(min(I1, [], 1), [], 2), [307 307]);
I1 = I1 ./ repmat(max(max(I1, [], 1), [], 2), [307 307]);
image = I1;

image_noisy = image + randn(size(I1)) * (100/255);
image_noisy(image_noisy < 0) = 0;
image_noisy(image_noisy > 1) = 1;
save('/home/panos/Desktop/HSI/new/deep-hs-prior/data/denoising/urban/denoising.mat', 'image', 'image_noisy');
imshow(cat(3, I1(:, :, 135), I1(:, :, 81), I1(:, :, 27)))

%%
% load("/home/panos/Desktop/HSI/new/deep-hs-prior/results/urban/result_denoising_2D_it01800.mat", "pred_avg");
% p = psnr(double(pred_avg), image(:, :, 1:191));
% s = ssim(double(pred_avg), image(:, :, 1:191));

% fprintf('psnr: %f, ssim: %f  \n', p, s);
% figure; imshow(cat(3, pred_avg(:, :, 135), pred_avg(:, :, 81), pred_avg(:, :, 27)));

%%
% imwrite(cat(3, pred_avg(:, :, 135), pred_avg(:, :, 81), pred_avg(:, :, 27)), 'urban.png');